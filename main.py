import argparse
import os
import yaml
import math
import json

import wandb
import torch
from tqdm import tqdm

import random
import numpy as np

from datasets import get_dvlog_dataloader, get_lmvd_dataloader
import torch.nn.functional as F
from models import DepMamba
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


CONFIG_PATH = "./config/config.yaml"
TEACHER_CKPT = "/home/ac/data/bai/DepMamba-main/Teacher_checkpoints/es_mamba_v1_teacher.pt"
USE_SELF_KD = True  # ⭐ True=用自蒸馏 KD；False=完全关掉 KD，当纯 supervised baseline

def get_lambda_kd(epoch: int, max_epoch: int) -> float:
    """
    KD 退火策略：
      - 前 25% epoch：KD 权重 = max_kd
      - 中间 50% epoch：线性从 max_kd 降到 0
      - 最后 25% epoch：KD = 0

    如果 USE_SELF_KD=False，则全程返回 0（完全不蒸馏）。
    """
    if not USE_SELF_KD:
        return 0.0

    warmup_ratio = 0.25
    decay_ratio  = 0.50

    warmup_end = int(max_epoch * warmup_ratio)
    decay_end  = int(max_epoch * (warmup_ratio + decay_ratio))

    max_kd = 0.5   # 自蒸馏阶段 KD 的最高权重，你方案 A 用的是 0.5 就继续沿用

    if epoch < warmup_end:
        return max_kd

    if epoch < decay_end:
        denom = max(1, decay_end - warmup_end)
        return max_kd * (1.0 - (epoch - warmup_end) / denom)

    return 0.0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     # 禁止自动寻找最优算子

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", type=bool)
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", type=bool)
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args


def train_epoch(
    net,
    teacher_net,                 # <<< 新增
    train_loader,
    loss_fn,
    optimizer,
    device,
    current_epoch,
    total_epochs,
    tqdm_able,
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.0
    correct_count = 0

    # --- 为画曲线准备的一些 epoch 级统计 ---
    gate_a_sum = 0.0
    gate_a_keep_sum = 0.0
    gate_v_sum = 0.0
    gate_v_keep_sum = 0.0
    gate_count = 0

    loss_s_sum = 0.0
    loss_c_sum = 0.0
    kd_loss_sum = 0.0          # <<< 新增：统计 KD loss

    # ====== ES-Mamba: 控制 EvidenceSelector 的 tau / hard ======
    core_net = net.module if isinstance(net, torch.nn.DataParallel) else net

    # 1) soft->hard curriculum：前 30 个 epoch 一律 hard=False
    hard_start_epoch = 99999
    use_hard = current_epoch >= hard_start_epoch

    # 2) tau 退火：从 3.0 慢慢降到 0.5，别再往下了
    tau0 = 3.0
    tau_min = 0.5
    tau_now = max(tau_min, tau0 * math.exp(-0.05 * current_epoch))

    # 更新 audio selector
    if hasattr(core_net, "audio_selector"):
        core_net.audio_selector.hard = use_hard
        core_net.audio_selector.tau = tau_now
    # =======================================================

    # KD 系数（这个 epoch 整体共用一份，带退火）
    lambda_kd = get_lambda_kd(current_epoch, total_epochs) if teacher_net is not None else 0.0

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=tqdm_able,
    ) as pbar:
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)

            # -------- Teacher / Student 前向（只用 logits 做 KD）--------
            if (teacher_net is not None) and (lambda_kd > 0.0):
                # Teacher 只做前向，不求梯度
                with torch.no_grad():
                    t_logits = teacher_net(x, mask)   # (B, 1)

                s_logits = net(x, mask)              # (B, 1)
            else:
                # 不做 KD，只算学生
                s_logits = net(x, mask)
                t_logits = None

            # 兼容 DataParallel 和 单卡两种情况
            core_net = net.module if isinstance(net, torch.nn.DataParallel) else net
            gate_a = getattr(core_net, "last_audio_gate", None)  # (B, 1, L) or None
            gate_v = getattr(core_net, "last_video_gate", None)  # (B, 1, L) or None

            # ====== audio gate：打印 + 正则 ======
            if gate_a is not None:
                g = gate_a                           # 不 detach，用来算正则
                g_det = g.detach()                   # 用来统计

                # 每 2 个 epoch、10% 的 batch 打一次
                if (current_epoch % 2 == 0) and (random.random() < 0.1):
                    keep_ratio = (g_det > 0.5).float().mean().item()
                    print(
                        "[audio] gate stats: mean={:.3f}, min={:.3f}, max={:.3f}, keep@0.5={:.3f}".format(
                            g_det.mean().item(),
                            g_det.min().item(),
                            g_det.max().item(),
                            keep_ratio,
                        )
                    )

                # 稀疏 + 连续性正则（目前只对 audio）
                loss_sparsity = g.mean()
                diff = torch.abs(g[..., 1:] - g[..., :-1])
                loss_continuity = diff.mean()
            else:
                loss_sparsity = torch.tensor(0.0, device=device)
                loss_continuity = torch.tensor(0.0, device=device)

            # ====== video gate：只打印，不进 loss ======
            if gate_v is not None and (current_epoch % 2 == 0) and (random.random() < 0.1):
                gv = gate_v.detach()
                keep_ratio_v = (gv > 0.5).float().mean().item()
                print(
                    "[video] gate stats: mean={:.3f}, min={:.3f}, max={:.3f}, keep@0.5={:.3f}".format(
                        gv.mean().item(),
                        gv.min().item(),
                        gv.max().item(),
                        keep_ratio_v,
                    )
                )

            lambda_sparse = get_lambda_sparse(current_epoch)
            lambda_cont = get_lambda_cont(current_epoch)
            loss_cls = loss_fn(s_logits, y.to(torch.float32))
            loss_s_term = lambda_sparse * loss_sparsity
            loss_c_term = lambda_cont * loss_continuity

            # ----- KD loss：只在 Teacher 预测正确的样本上蒸馏 -----
            if (teacher_net is not None) and (lambda_kd > 0.0) and (t_logits is not None):
                # teacher / student 的 sigmoid 概率
                p_t = torch.sigmoid(t_logits.detach())   # (B, 1)
                p_s = torch.sigmoid(s_logits)            # (B, 1)

                # 每个样本的 MSE
                kd_per_sample = F.mse_loss(p_s, p_t, reduction='none')   # (B, 1)
                kd_per_sample = kd_per_sample.view(-1)                   # (B,)

                # Teacher 的 hard 预测
                pred_t = (t_logits > 0.0).int()          # (B, 1)

                # GT 标签（确保是 int）
                targets = y.int()                         # (B, 1)

                # 哪些样本 Teacher 预测正确
                is_correct = (pred_t == targets).view(-1).float()   # (B,)

                if is_correct.sum() > 0:
                    # 只对 Teacher 正确的样本求平均
                    loss_kd = (kd_per_sample * is_correct).sum() / is_correct.sum()
                else:
                    loss_kd = torch.tensor(0.0, device=device)
            else:
                loss_kd = torch.tensor(0.0, device=device)

            if current_epoch % 2 == 0 and random.random() < 0.05:
                print(
                    f"[epoch {current_epoch}] "
                    f"loss_cls={loss_cls.item():.3f}, "
                    f"λ_s*Ls={loss_s_term.item():.3f}, "
                    f"λ_c*Lc={loss_c_term.item():.3f}, "
                    f"λ_kd*L_kd={(lambda_kd * loss_kd.item()):.3f}"
                )

            # 总 loss
            loss = loss_cls + loss_s_term + loss_c_term + lambda_kd * loss_kd
            loss.backward()

            # ------- 统计 gate / loss_s / loss_c / loss_kd 的 epoch 均值 -------
            bsz = x.size(0)
            if gate_a is not None:
                g_det = gate_a.detach()
                gate_a_sum += g_det.mean().item() * bsz
                gate_a_keep_sum += (g_det > 0.5).float().mean().item() * bsz
                loss_s_sum += loss_s_term.item() * bsz
                loss_c_sum += loss_c_term.item() * bsz
                gate_count += bsz

            if gate_v is not None:
                gv_det = gate_v.detach()
                gate_v_sum += gv_det.mean().item() * bsz
                gate_v_keep_sum += (gv_det > 0.5).float().mean().item() * bsz

            kd_loss_sum += (lambda_kd * loss_kd.item()) * bsz

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            # binary classification with only one output neuron
            pred = (s_logits > 0.0).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / sample_count,
                    "acc": correct_count / sample_count,
                }
            )

    epoch_loss = running_loss / sample_count
    epoch_acc = correct_count / sample_count

    if gate_count > 0:
        gate_a_mean = gate_a_sum / gate_count
        gate_a_keep = gate_a_keep_sum / gate_count
        gate_v_mean = gate_v_sum / gate_count
        gate_v_keep = gate_v_keep_sum / gate_count
        loss_s_avg = loss_s_sum / gate_count
        loss_c_avg = loss_c_sum / gate_count
    else:
        gate_a_mean = gate_a_keep = 0.0
        gate_v_mean = gate_v_keep = 0.0
        loss_s_avg = loss_c_avg = 0.0

    loss_kd_avg = kd_loss_sum / sample_count if sample_count > 0 else 0.0

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "gate_a_mean": gate_a_mean,
        "gate_a_keep": gate_a_keep,
        "gate_v_mean": gate_v_mean,
        "gate_v_keep": gate_v_keep,
        "loss_s": loss_s_avg,
        "loss_c": loss_c_avg,
        "loss_kd": loss_kd_avg,     # <<< 新增
    }


def val(
    net, val_loader, loss_fn, device, tqdm_able
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                # print(x.shape,y.shape)
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }

def eval_with_scores(net, data_loader, loss_fn, device, tqdm_able):
    """
    和 val 类似，但额外返回：
      - y_true: 所有样本的 0/1 标签 (numpy array)
      - y_score: 所有样本的正类概率（sigmoid(logit））(numpy array)
    用来画 ROC / 计算 AUC.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    all_labels = []
    all_scores = []

    with torch.no_grad():
        with tqdm(
            data_loader, desc="Evaluating (with scores)", leave=False,
            unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                logits = net(x, mask)                         # (B,1)
                loss = loss_fn(logits, y.to(torch.float32))

                # ======== 收集用于 ROC 的分数和标签 ========
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()  # 正类概率
                labels = y.view(-1).cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels)
                # ===========================================

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                pred = (logits > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )

    metrics = {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    return metrics, all_labels, all_scores

# ====== ES-Mamba: 稀疏 / 连续性正则的 warm-up 系数 ======
def get_lambda_sparse(epoch: int) -> float:
    """
    稀疏正则：前 30 epoch 不加，
    30~90 线性升到 0.1，之后保持 0.1。
    """
    if epoch < 30:
        return 0.0
    elif epoch < 90:
        return 0.1 * (epoch - 30) / 60.0   # 0 -> 0.1
    else:
        return 0.1


def get_lambda_cont(epoch: int) -> float:
    """
    连续性正则：力度再小一点，只做“平滑”，别主导训练。
    """
    if epoch < 30:
        return 0.0
    else:
        return 0.01
# ========================================================

def main():
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir,args.dataset)
    for i_iter in range(3):
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_loss_s": [],
            "train_loss_c": [],
            "gate_a_mean": [],
            "gate_a_keep": [],
            "gate_v_mean": [],
            "gate_v_keep": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "train_loss_kd": [],
        }
        if args.if_wandb:
            wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
            wandb.init(
                project="mamnba_ad", config=args, name=wandb_run_name,
            )
            args = wandb.config
        print(args)
        # Build Save Dir
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/samples", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints", exist_ok=True)

        # construct the model
        # ---------- construct the student model (ES-DepMamba, use_gate=True) ----------
        if args.model == "DepMamba":
            if args.dataset == "lmvd":
                student_cfg = dict(args.mmmamba_lmvd)
            elif args.dataset == "dvlog":
                student_cfg = dict(args.mmmamba)
            else:
                raise ValueError(f"Unknown dataset {args.dataset}")

            student_cfg["use_gate"] = True   # 学生：打开 gate
            net = DepMamba(**student_cfg)
        else:
            raise NotImplementedError(
                f"The {args.model} method has not been implemented by this repo"
            )

        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        # ---------- construct the teacher model (frozen, ES-Mamba V1) ----------
        teacher_net = None
        if USE_SELF_KD and os.path.exists(TEACHER_CKPT):
            print(f"Loading ES-Mamba V1 teacher checkpoint from: {TEACHER_CKPT}")

            # 和 student 一样的配置（包括 use_gate=True），确保结构完全一致
            if args.dataset == "lmvd":
                teacher_cfg = dict(args.mmmamba_lmvd)
            elif args.dataset == "dvlog":
                teacher_cfg = dict(args.mmmamba)
            else:
                raise ValueError(f"Unknown dataset {args.dataset}")

            teacher_cfg["use_gate"] = True      # V1 当时就是打开 gate 训练的

            teacher_net = DepMamba(**teacher_cfg)
            teacher_net = teacher_net.to(args.device[0])

            state = torch.load(TEACHER_CKPT, map_location=args.device[0])
            teacher_net.load_state_dict(state, strict=True)
            print("ES-Mamba V1 teacher ckpt loaded successfully.")

            for p in teacher_net.parameters():
                p.requires_grad = False
            teacher_net.eval()
        else:
            print(f"[WARN] Teacher checkpoint not found at {TEACHER_CKPT}，本次训练不做 KD。")

        # prepare the data
        if args.dataset=='dvlog':
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset=='lmvd':
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

        # set other training components
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0
        best_test_acc = -1.0
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net,teacher_net, train_loader, loss_fn, optimizer, 
                    args.device[0], epoch, args.epochs, args.tqdm_able
                )
                val_results = val(net, val_loader, loss_fn, args.device[0],args.tqdm_able)
                 # ---- 记录曲线用的指标 ----
                history["train_loss"].append(float(train_results["loss"]))
                history["train_acc"].append(float(train_results["acc"]))
                history["train_loss_s"].append(float(train_results["loss_s"]))
                history["train_loss_c"].append(float(train_results["loss_c"]))
                history["train_loss_kd"].append(float(train_results["loss_kd"]))  # <<< 新增
                history["gate_a_mean"].append(float(train_results["gate_a_mean"]))
                history["gate_a_keep"].append(float(train_results["gate_a_keep"]))
                history["gate_v_mean"].append(float(train_results["gate_v_mean"]))
                history["gate_v_keep"].append(float(train_results["gate_v_keep"]))

                history["val_loss"].append(float(val_results["loss"]))
                history["val_acc"].append(float(val_results["acc"]))
                history["val_precision"].append(float(val_results["precision"]))
                history["val_recall"].append(float(val_results["recall"]))
                history["val_f1"].append(float(val_results["f1"]))

                val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(net.state_dict(),f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt")

                if args.if_wandb:
                    wandb.log({
                        "loss/train": train_results["loss"],
                        "acc/train": train_results["acc"],
                        "loss/val": val_results["loss"],
                        "acc/val": val_results["acc"],
                        "precision/val": val_results["precision"],
                        "recall/val": val_results["recall"],
                        "f1/val": val_results["f1"]
                    })
            
        # upload the best model to wandb website
        # load the best model for testing
        with torch.no_grad():
            net.load_state_dict(
                torch.load(
                    f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt",
                    map_location=args.device[0],
                )
            )
            net.eval()

            # ===== 用带 scores 的评估函数 =====
            test_results, y_true, y_score = eval_with_scores(
                net, test_loader, loss_fn, args.device[0], args.tqdm_able
            )
            print("Test results:")
            print(test_results)

            # ===== 计算 ROC 曲线和 AUC =====
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            print(f"AUC (test) = {roc_auc:.4f}")

            # 保存 ROC 数据，方便之后画多个模型的对比图
            run_dir = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}"
            os.makedirs(run_dir, exist_ok=True)
            np.savez(
                os.path.join(run_dir, "roc_test.npz"),
                fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc
            )

            # 画单条 ROC 曲线（当前这个模型）
            plt.figure()
            plt.plot(fpr, tpr, label=f"{args.model} (AUC={roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC on {args.dataset} (test)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_test.png"), dpi=300)
            plt.close()

            # -------- 保存 test 结果到 txt --------
            avg_score = (
                test_results["acc"]
                + test_results["precision"]
                + test_results["recall"]
                + test_results["f1"]
            ) / 4.0

            results_path = f'./results/{args.dataset}_{args.model}_{str(i_iter)}.txt'
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                test_result_str = (
                    f'Accuracy:{test_results["acc"]}, '
                    f'Precision:{test_results["precision"]}, '
                    f'Recall:{test_results["recall"]}, '
                    f'F1:{test_results["f1"]}, '
                    f'Avg:{avg_score}'
                )
                f.write(test_result_str)

            # -------- 保存本次 run 的曲线数据（放到当前 run 的目录里）--------
            run_dir = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}"
            curve_path = os.path.join(run_dir, "curves.json")
            with open(curve_path, "w") as f_json:
                json.dump(history, f_json, indent=2)
            print("Curve stats saved to:", curve_path, flush=True)


    if args.if_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(f"{args.save_dir}/{args.model}/checkpoints/best_model.pt")
        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()


if __name__ == '__main__':
    setup_seed(2222)
    main()
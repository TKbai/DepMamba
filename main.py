import argparse
import os
import yaml
import json
import inspect
import random
import math

import wandb
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader


CONFIG_PATH = "./config/config.yaml"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Train and test a model.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", type=str2bool)
    parser.add_argument("-tqdm", "--tqdm_able", type=str2bool)
    parser.add_argument("-tr", "--train", type=str2bool)
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def get_core_model(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def sanitize_depmamba_cfg(cfg: dict):
    """
    兼容旧 config 和新 DepMamba.__init__ 的字段。
    如果新模型没有某些旧字段，就自动忽略。
    """
    cfg = dict(cfg)

    # 兼容旧字段名
    if "mm_output_sizes" in cfg and "uni_output_sizes" not in cfg:
        cfg["uni_output_sizes"] = cfg["mm_output_sizes"]

    # 如果你想让 fusion_output_sizes 也可由 config 控制，这里优先保留 config 原值
    if "fusion_output_sizes" not in cfg:
        cfg["fusion_output_sizes"] = [128]

    # 默认打开 gate
    if "use_gate" not in cfg:
        cfg["use_gate"] = False

    # selector / attention 默认值
    cfg.setdefault("selector_tau", 1.0)
    cfg.setdefault("selector_hard", False)
    cfg.setdefault("selector_alpha", 0.5)
    cfg.setdefault("attn_heads", 1)

    # 过滤掉新 DepMamba 不接受的参数
    valid_params = set(inspect.signature(DepMamba.__init__).parameters.keys())
    valid_params.discard("self")
    cfg = {k: v for k, v in cfg.items() if k in valid_params}

    return cfg


def get_lambda_sparse(epoch: int) -> float:
    if epoch < 30:
        return 0.0
    elif epoch < 90:
        return 0.05 * (epoch - 30) / 60.0
    else:
        return 0.05


def get_lambda_cont(epoch: int) -> float:
    if epoch < 30:
        return 0.0
    else:
        return 0.005

def get_lambda_mil(epoch: int) -> float:
    # 先 warm-up，避免一开始 selector 还没成型就被 proposal-style loss 拉偏
    if epoch < 10:
        return 0.0
    elif epoch < 30:
        return 0.05
    else:
        return 0.1


def get_lambda_comp_aux(epoch: int) -> float:
    if epoch < 10:
        return 0.0
    else:
        return 0.02


def masked_topk_mean(score_map, valid_mask=None, topk_ratio=0.1):
    """
    score_map: (B,1,L)
    valid_mask: (B,L), 1=valid
    return: (B,1)
    """
    s = score_map.squeeze(1)   # (B,L)
    B, L = s.shape
    bag_scores = []

    for i in range(B):
        if valid_mask is None:
            cur = s[i]
        else:
            cur = s[i][valid_mask[i].bool()]

        if cur.numel() == 0:
            bag_scores.append(torch.zeros((), device=s.device, dtype=s.dtype))
            continue

        k = max(1, int(math.ceil(cur.numel() * topk_ratio)))
        topk_vals = torch.topk(cur, k=k, dim=-1).values
        bag_scores.append(topk_vals.mean())

    return torch.stack(bag_scores, dim=0).unsqueeze(1)  # (B,1)


def topk_mil_loss(score_map, labels, valid_mask=None, topk_ratio=0.1):
    """
    用 evidence score 的 top-k 聚合做 bag-level BCE
    """
    bag_logits = masked_topk_mean(score_map, valid_mask, topk_ratio)
    return F.binary_cross_entropy_with_logits(bag_logits, labels.float())


def completeness_aux_loss(raw_score, comp_score, labels, valid_mask=None, topk_ratio=0.1):
    """
    轻量 completeness:
    只在高 evidence 区域强调 completeness
    """
    # 用 raw evidence 的 sigmoid 作为权重，detach 避免互相拖拽太厉害
    weighted_comp = comp_score * torch.sigmoid(raw_score.detach())
    bag_logits = masked_topk_mean(weighted_comp, valid_mask, topk_ratio)
    return F.binary_cross_entropy_with_logits(bag_logits, labels.float())


def train_epoch(
    net,
    train_loader,
    loss_fn,
    optimizer,
    device,
    current_epoch,
    total_epochs,
    tqdm_able,
):
    net.train()
    core_net = get_core_model(net)

    # 新版本 selector 默认走确定性 soft gate
    if hasattr(core_net, "audio_selector") and hasattr(core_net.audio_selector, "use_gumbel"):
        core_net.audio_selector.use_gumbel = False
    if hasattr(core_net, "video_selector") and hasattr(core_net.video_selector, "use_gumbel"):
        core_net.video_selector.use_gumbel = False

    sample_count = 0
    running_loss = 0.0
    correct_count = 0

    gate_a_sum = 0.0
    gate_a_keep_sum = 0.0
    gate_v_sum = 0.0
    gate_v_keep_sum = 0.0
    gate_count = 0

    loss_s_sum = 0.0
    loss_c_sum = 0.0

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=not tqdm_able,
    ) as pbar:
        for x, y, mask in pbar:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            mask = mask.to(device)
            

            optimizer.zero_grad(set_to_none=True)

            # 新版 DepMamba 仍支持 return_feat=True -> (logits, feat)
            s_logits, s_feat = net(x, mask, return_feat=True)

            core_net = get_core_model(net)
            gate_a = getattr(core_net, "last_audio_gate", None)  # (B,1,T) or None
            gate_v = getattr(core_net, "last_video_gate", None)  # (B,1,T) or None
            
            aux_a = None
            aux_v = None

            if getattr(core_net, "audio_selector", None) is not None:
                aux_a = getattr(core_net.audio_selector, "last_aux", None)

            if getattr(core_net, "video_selector", None) is not None:
                aux_v = getattr(core_net.video_selector, "last_aux", None)

            # ====== 对称 gate 正则 ======
            loss_sparsity = torch.tensor(0.0, device=device)
            loss_continuity = torch.tensor(0.0, device=device)

            num_gate_terms = 0

            if gate_a is not None:
                ga = gate_a
                ga_det = ga.detach()



                loss_sparsity = loss_sparsity + ga.mean()
                loss_continuity = loss_continuity + torch.abs(ga[..., 1:] - ga[..., :-1]).mean()
                num_gate_terms += 1

            if gate_v is not None:
                gv = gate_v
                gv_det = gv.detach()



                loss_sparsity = loss_sparsity + gv.mean()
                loss_continuity = loss_continuity + torch.abs(gv[..., 1:] - gv[..., :-1]).mean()
                num_gate_terms += 1

            if num_gate_terms > 0:
                loss_sparsity = loss_sparsity / num_gate_terms
                loss_continuity = loss_continuity / num_gate_terms
            else:
                loss_sparsity = torch.tensor(0.0, device=device)
                loss_continuity = torch.tensor(0.0, device=device)

            lambda_sparse = get_lambda_sparse(current_epoch)
            lambda_cont = get_lambda_cont(current_epoch)
            lambda_mil = get_lambda_mil(current_epoch)
            lambda_comp_aux = get_lambda_comp_aux(current_epoch)

            loss_cls = loss_fn(s_logits, y.to(torch.float32))
            loss_s_term = lambda_sparse * loss_sparsity
            loss_c_term = lambda_cont * loss_continuity

            # ===== proposal-aware ASG v2: top-k MIL + completeness aux =====
            loss_mil = torch.tensor(0.0, device=device)
            loss_comp_aux = torch.tensor(0.0, device=device)
            num_aux_terms = 0

            if aux_a is not None and "final_score" in aux_a:
                loss_mil = loss_mil + topk_mil_loss(
                    aux_a["final_score"], y, mask, topk_ratio=0.1
                )
                loss_comp_aux = loss_comp_aux + completeness_aux_loss(
                    aux_a["raw_score"], aux_a["comp_score"], y, mask, topk_ratio=0.1
                )
                num_aux_terms += 1

            if aux_v is not None and "final_score" in aux_v:
                loss_mil = loss_mil + topk_mil_loss(
                    aux_v["final_score"], y, mask, topk_ratio=0.1
                )
                loss_comp_aux = loss_comp_aux + completeness_aux_loss(
                    aux_v["raw_score"], aux_v["comp_score"], y, mask, topk_ratio=0.1
                )
                num_aux_terms += 1

            if num_aux_terms > 0:
                loss_mil = loss_mil / num_aux_terms
                loss_comp_aux = loss_comp_aux / num_aux_terms
            else:
                loss_mil = torch.tensor(0.0, device=device)
                loss_comp_aux = torch.tensor(0.0, device=device)

            loss_mil_term = lambda_mil * loss_mil
            loss_comp_term = lambda_comp_aux * loss_comp_aux

            loss = loss_cls + loss_s_term + loss_c_term + loss_mil_term + loss_comp_term

            if current_epoch % 2 == 0 and random.random() < 0.05:
                print(
                    f"[epoch {current_epoch}] "
                    f"loss_cls={loss_cls.item():.3f}, "
                    f"λ_s*Ls={loss_s_term.item():.3f}, "
                    f"λ_c*Lc={loss_c_term.item():.3f}, "
                    f"λ_mil*Lm={loss_mil_term.item():.3f}, "
                    f"λ_comp*Lcomp={loss_comp_term.item():.3f}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # ------- 统计 -------
            bsz = x.size(0)

            if gate_a is not None:
                ga_det = gate_a.detach()
                gate_a_sum += ga_det.mean().item() * bsz
                gate_a_keep_sum += (ga_det > 0.5).float().mean().item() * bsz
                gate_count += bsz

            if gate_v is not None:
                gv_det = gate_v.detach()
                gate_v_sum += gv_det.mean().item() * bsz
                gate_v_keep_sum += (gv_det > 0.5).float().mean().item() * bsz

            loss_s_sum += loss_s_term.item() * bsz
            loss_c_sum += loss_c_term.item() * bsz

            sample_count += bsz
            running_loss += loss.item() * bsz

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
    else:
        gate_a_mean = gate_a_keep = 0.0
        gate_v_mean = gate_v_keep = 0.0

    loss_s_avg = loss_s_sum / sample_count if sample_count > 0 else 0.0
    loss_c_avg = loss_c_sum / sample_count if sample_count > 0 else 0.0

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "gate_a_mean": gate_a_mean,
        "gate_a_keep": gate_a_keep,
        "gate_v_mean": gate_v_mean,
        "gate_v_keep": gate_v_keep,
        "loss_s": loss_s_avg,
        "loss_c": loss_c_avg,
        "loss_mil": loss_mil_term.item() if sample_count > 0 else 0.0,
        "loss_comp_aux": loss_comp_term.item() if sample_count > 0 else 0.0,
    }


def val(net, val_loader, loss_fn, device, tqdm_able):
    net.eval()
    sample_count = 0
    running_loss = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader,
            desc="Validating",
            leave=False,
            unit="batch",
            disable=not tqdm_able,
        ) as pbar:
            for x, y, mask in pbar:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                mask = mask.to(device)

                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                pred = (y_pred > 0.0).int()
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
                accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0

                pbar.set_postfix(
                    {
                        "loss": l,
                        "acc": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1_score,
                    }
                )

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0
    return {
        "loss": l,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def main():
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    last_best_ckpt_path = None

    for i_iter in range(3):
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_loss_s": [],
            "train_loss_c": [],
            "train_loss_mil": [],
            "train_loss_comp_aux": [],
            "gate_a_mean": [],
            "gate_a_keep": [],
            "gate_v_mean": [],
            "gate_v_keep": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        if args.if_wandb:
            wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
            wandb.init(project="mamnba_ad", config=args, name=wandb_run_name)
            args = wandb.config

        print(args)

        run_dir = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}"
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(f"{run_dir}/samples", exist_ok=True)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

        # ===== construct model =====
        if args.model == "DepMamba":
            if args.dataset == "lmvd":
                student_cfg = dict(args.mmmamba_lmvd)
            elif args.dataset == "dvlog":
                student_cfg = dict(args.mmmamba)
            else:
                raise ValueError(f"Unknown dataset {args.dataset}")

            student_cfg = sanitize_depmamba_cfg(student_cfg)
            net = DepMamba(**student_cfg)
        else:
            raise NotImplementedError(
                f"The {args.model} method has not been implemented by this repo"
            )

        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        # ===== prepare data =====
        if args.dataset == "dvlog":
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset == "lmvd":
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0

        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net,
                    train_loader,
                    loss_fn,
                    optimizer,
                    args.device[0],
                    epoch,
                    args.epochs,
                    args.tqdm_able,
                )
                val_results = val(net, val_loader, loss_fn, args.device[0], args.tqdm_able)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_results['loss']:.4f}, "
                    f"train_acc={train_results['acc']:.4f}, "
                    f"val_loss={val_results['loss']:.4f}, "
                    f"val_acc={val_results['acc']:.4f}, "
                    f"val_f1={val_results['f1']:.4f}"
                )

                history["train_loss"].append(float(train_results["loss"]))
                history["train_acc"].append(float(train_results["acc"]))
                history["train_loss_s"].append(float(train_results["loss_s"]))
                history["train_loss_c"].append(float(train_results["loss_c"]))
                history["train_loss_mil"].append(float(train_results["loss_mil"]))
                history["train_loss_comp_aux"].append(float(train_results["loss_comp_aux"]))
                history["gate_a_mean"].append(float(train_results["gate_a_mean"]))
                history["gate_a_keep"].append(float(train_results["gate_a_keep"]))
                history["gate_v_mean"].append(float(train_results["gate_v_mean"]))
                history["gate_v_keep"].append(float(train_results["gate_v_keep"]))

                history["val_loss"].append(float(val_results["loss"]))
                history["val_acc"].append(float(val_results["acc"]))
                history["val_precision"].append(float(val_results["precision"]))
                history["val_recall"].append(float(val_results["recall"]))
                history["val_f1"].append(float(val_results["f1"]))

                val_acc = (
                    val_results["acc"]
                    + val_results["precision"]
                    + val_results["recall"]
                    + val_results["f1"]
                ) / 4.0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_ckpt_path = f"{run_dir}/checkpoints/best_model.pt"
                    torch.save(get_core_model(net).state_dict(), best_ckpt_path)
                    last_best_ckpt_path = best_ckpt_path

                if args.if_wandb:
                    wandb.log(
                        {
                            "loss/train": train_results["loss"],
                            "acc/train": train_results["acc"],
                            "loss/train_s": train_results["loss_s"],
                            "loss/train_c": train_results["loss_c"],
                            "gate/audio_mean": train_results["gate_a_mean"],
                            "gate/audio_keep": train_results["gate_a_keep"],
                            "gate/video_mean": train_results["gate_v_mean"],
                            "gate/video_keep": train_results["gate_v_keep"],
                            "loss/val": val_results["loss"],
                            "acc/val": val_results["acc"],
                            "precision/val": val_results["precision"],
                            "recall/val": val_results["recall"],
                            "f1/val": val_results["f1"],
                        }
                    )

        # ===== load best model for testing =====
        best_ckpt_path = f"{run_dir}/checkpoints/best_model.pt"
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

        core_net = get_core_model(net)
        core_net.load_state_dict(torch.load(best_ckpt_path, map_location=args.device[0]))
        core_net.eval()

        with torch.no_grad():
            test_results = val(net, test_loader, loss_fn, args.device[0], args.tqdm_able)
            print("Test results:")
            print(test_results)

            avg_score = (
                test_results["acc"]
                + test_results["precision"]
                + test_results["recall"]
                + test_results["f1"]
            ) / 4.0

            results_path = f"./results/{args.dataset}_{args.model}_{str(i_iter)}.txt"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                test_result_str = (
                    f'Accuracy:{test_results["acc"]}, '
                    f'Precision:{test_results["precision"]}, '
                    f'Recall:{test_results["recall"]}, '
                    f'F1:{test_results["f1"]}, '
                    f"Avg:{avg_score}"
                )
                f.write(test_result_str)

            curve_path = os.path.join(run_dir, "curves.json")
            with open(curve_path, "w") as f_json:
                json.dump(history, f_json, indent=2)
            print("Curve stats saved to:", curve_path, flush=True)

    if args.if_wandb:
        if last_best_ckpt_path is not None and os.path.exists(last_best_ckpt_path):
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(last_best_ckpt_path)
            wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    setup_seed(2222)
    main()
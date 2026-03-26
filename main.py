
import argparse
import inspect
import json
import math
import os
import random
from typing import Dict, Optional
from torch.utils.data import DataLoader, Subset

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm

from models import DepMamba
from datasets import (
    get_dvlog_dataloader,
    get_lmvd_dataloader,
    get_search_dataloader,
)

CONFIG_PATH = "./config/config.yaml"


def setup_seed(seed: int):
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

    parser = argparse.ArgumentParser(description="Train and test DepMamba.")
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
    Keep backward compatibility with older configs.
    """
    cfg = dict(cfg)

    if "mm_output_sizes" in cfg and "uni_output_sizes" not in cfg:
        cfg["uni_output_sizes"] = cfg["mm_output_sizes"]

    if "fusion_output_sizes" in cfg and "uni_output_sizes" not in cfg:
        pass

    valid_params = set(inspect.signature(DepMamba.__init__).parameters.keys())
    valid_params.discard("self")
    cfg = {k: v for k, v in cfg.items() if k in valid_params}
    return cfg


def get_dataset_train_cfg(args):
    if args.dataset == "lmvd":
        return getattr(args, "lmvd_train", {})
    elif args.dataset == "dvlog":
        return getattr(args, "dvlog_train", {})
    elif args.dataset == "search":
        return getattr(args, "search_train", {})
    return {}


def get_model_cfg(args):
    if args.dataset == "lmvd":
        return dict(args.mmmamba_lmvd)
    elif args.dataset == "dvlog":
        return dict(args.mmmamba)
    elif args.dataset == "search":
        return dict(args.mmmamba_search)
    raise ValueError(f"Unknown dataset {args.dataset}")


def build_dataloaders(args):
    train_cfg = get_dataset_train_cfg(args)
    aug = bool(train_cfg.get("aug", False))

    if args.dataset == "dvlog":
        train_loader = get_dvlog_dataloader(
            args.data_dir, "train", args.batch_size, args.train_gender, aug=aug
        )
        val_loader = get_dvlog_dataloader(
            args.data_dir, "valid", args.batch_size, args.test_gender, aug=False
        )
        test_loader = get_dvlog_dataloader(
            args.data_dir, "test", args.batch_size, args.test_gender, aug=False
        )
    elif args.dataset == "lmvd":
        train_loader = get_lmvd_dataloader(
            args.data_dir, "train", args.batch_size, args.train_gender, aug=aug
        )
        val_loader = get_lmvd_dataloader(
            args.data_dir, "valid", args.batch_size, args.test_gender, aug=False
        )
        test_loader = get_lmvd_dataloader(
            args.data_dir, "test", args.batch_size, args.test_gender, aug=False
        )
    elif args.dataset == "search":
        train_loader = get_search_dataloader(
            args.data_dir, "train", args.batch_size, args.train_gender
        )
        val_loader = get_search_dataloader(
            args.data_dir, "valid", args.batch_size, args.test_gender
        )
        test_loader = get_search_dataloader(
            args.data_dir, "test", args.batch_size, args.test_gender
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    return train_loader, val_loader, test_loader


def masked_mean_1d(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        return x.mean(dim=-1)
    m = mask.float()
    return (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


def masked_tv_1d(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    diff = torch.abs(x[:, 1:] - x[:, :-1])
    if mask is None:
        return diff.mean(dim=-1)
    edge_mask = (mask[:, 1:] * mask[:, :-1]).float()
    return (diff * edge_mask).sum(dim=-1) / edge_mask.sum(dim=-1).clamp(min=1.0)


def masked_topk_comp_loss(
    score: torch.Tensor,
    contrast: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.15,
    topk_ratio: float = 0.1,
):
    """
    score, contrast: (B, T)
    labels: (B, 1)
    """
    B = score.size(0)
    pos_mask = labels.squeeze(1) > 0.5
    losses = []

    for b in range(B):
        if not bool(pos_mask[b].item()):
            continue

        if mask is None:
            cur_s = score[b]
            cur_c = contrast[b]
        else:
            valid = mask[b].bool()
            cur_s = score[b][valid]
            cur_c = contrast[b][valid]

        if cur_s.numel() == 0:
            continue

        k = max(1, int(math.ceil(cur_s.numel() * topk_ratio)))
        idx = torch.topk(cur_s, k=k, dim=-1).indices
        cur_loss = F.relu(margin - cur_c[idx]).mean()
        losses.append(cur_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=score.device, dtype=score.dtype)
    return torch.stack(losses).mean()


def compute_structural_loss(
    refined_score: torch.Tensor,   # (B, T)
    contrast: torch.Tensor,        # (B, T)
    labels: torch.Tensor,          # (B, 1)
    mask: Optional[torch.Tensor],
    cfg: Dict,
):
    mean_score = masked_mean_1d(refined_score, mask)
    tv_per_sample = masked_tv_1d(refined_score, mask)

    pos_mask = (labels.squeeze(1) > 0.5).float()
    neg_mask = 1.0 - pos_mask

    loss_neg = (mean_score * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
    loss_budget = ((F.relu(mean_score - cfg["budget_rho"]) ** 2) * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
    loss_tv = tv_per_sample.mean()
    loss_comp = masked_topk_comp_loss(
        refined_score,
        contrast,
        labels,
        mask=mask,
        margin=cfg["comp_margin"],
        topk_ratio=cfg["comp_topk_ratio"],
    )

    total = (
        cfg["lambda_neg"] * loss_neg
        + cfg["lambda_budget"] * loss_budget
        + cfg["lambda_tv"] * loss_tv
        + cfg["lambda_comp"] * loss_comp
    )

    return total, {
        "neg": float(loss_neg.detach().item()),
        "budget": float(loss_budget.detach().item()),
        "tv": float(loss_tv.detach().item()),
        "comp": float(loss_comp.detach().item()),
    }


def summarize_binary_counts(TP: int, FP: int, TN: int, FN: int):
    total = TP + FP + TN + FN
    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    pred_pos_rate = (TP + FP) / total if total > 0 else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "pred_pos_rate": pred_pos_rate,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


def evaluate_constant_baseline(data_loader, predict_positive: bool = True):
    TP, FP, TN, FN = 0, 0, 0, 0
    constant = 1 if predict_positive else 0
    with torch.no_grad():
        for _, y, _ in data_loader:
            y = y.view(-1).int()
            pred = torch.full_like(y, fill_value=constant)
            TP += torch.sum((pred == 1) & (y == 1)).item()
            FP += torch.sum((pred == 1) & (y == 0)).item()
            TN += torch.sum((pred == 0) & (y == 0)).item()
            FN += torch.sum((pred == 0) & (y == 1)).item()
    stats = summarize_binary_counts(TP, FP, TN, FN)
    stats["name"] = "all_positive" if predict_positive else "all_negative"
    return stats


def make_balanced_subset_loader(loader, pos_count: int = 8, neg_count: int = 8, shuffle: bool = True):
    dataset = loader.dataset
    if hasattr(dataset, "labels"):
        labels = [int(v) for v in dataset.labels]
    else:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]

    pos_idx = [i for i, y in enumerate(labels) if y == 1][:pos_count]
    neg_idx = [i for i, y in enumerate(labels) if y == 0][:neg_count]
    indices = pos_idx + neg_idx
    if len(indices) == 0:
        raise RuntimeError("Balanced subset is empty.")

    subset = Subset(dataset, indices)
    bs = min(loader.batch_size or len(indices), len(indices))

    return DataLoader(
        subset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=loader.collate_fn,
        num_workers=0,
        drop_last=False,
    )


def get_lambda_struct(epoch: int, total_epochs: int, peak: float):
    warmup_epochs = min(10, max(3, total_epochs // 12))
    if epoch < warmup_epochs:
        return peak * float(epoch + 1) / float(warmup_epochs)
    return peak


def train_epoch(
    net,
    train_loader,
    loss_fn,
    optimizer,
    device,
    current_epoch,
    total_epochs,
    tqdm_able,
    struct_cfg: Dict,
):
    net.train()

    sample_count = 0
    running_loss = 0.0
    running_cls = 0.0
    running_struct = 0.0
    running_neg = 0.0
    running_budget = 0.0
    running_tv = 0.0
    running_comp = 0.0

    TP, FP, TN, FN = 0, 0, 0, 0
    logit_sum = 0.0
    logit_sq_sum = 0.0
    prob_sum = 0.0

    lambda_struct = get_lambda_struct(
        current_epoch,
        total_epochs,
        peak=float(struct_cfg["lambda_struct_peak"]),
    )

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=not tqdm_able,
    ) as pbar:
        for x, y, mask in pbar:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            info = net(x, padding_mask=mask, return_intermediate=True)
            logits = info["logits"]

            loss_cls = loss_fn(logits, y)

            loss_struct_a, parts_a = compute_structural_loss(
                refined_score=info["audio_refined"],
                contrast=info["audio_contrast"],
                labels=y,
                mask=mask,
                cfg=struct_cfg,
            )
            loss_struct_v, parts_v = compute_structural_loss(
                refined_score=info["video_refined"],
                contrast=info["video_contrast"],
                labels=y,
                mask=mask,
                cfg=struct_cfg,
            )

            loss_struct = 0.5 * (loss_struct_a + loss_struct_v)
            loss = loss_cls + lambda_struct * loss_struct

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            bsz = x.size(0)
            sample_count += bsz
            running_loss += loss.item() * bsz
            running_cls += loss_cls.item() * bsz
            running_struct += (lambda_struct * loss_struct).item() * bsz
            running_neg += 0.5 * (parts_a["neg"] + parts_v["neg"]) * bsz
            running_budget += 0.5 * (parts_a["budget"] + parts_v["budget"]) * bsz
            running_tv += 0.5 * (parts_a["tv"] + parts_v["tv"]) * bsz
            running_comp += 0.5 * (parts_a["comp"] + parts_v["comp"]) * bsz

            pred = (logits > 0.0).int()
            y_int = y.int()
            TP += torch.sum((pred == 1) & (y_int == 1)).item()
            FP += torch.sum((pred == 1) & (y_int == 0)).item()
            TN += torch.sum((pred == 0) & (y_int == 0)).item()
            FN += torch.sum((pred == 0) & (y_int == 1)).item()

            logit_sum += logits.detach().sum().item()
            logit_sq_sum += (logits.detach() ** 2).sum().item()
            prob_sum += torch.sigmoid(logits.detach()).sum().item()

            live_stats = summarize_binary_counts(TP, FP, TN, FN)
            pbar.set_postfix(
                {
                    "loss": running_loss / sample_count,
                    "bal_acc": live_stats["balanced_acc"],
                    "pred_pos": live_stats["pred_pos_rate"],
                    "λ_struct": lambda_struct,
                }
            )

    stats = summarize_binary_counts(TP, FP, TN, FN)
    logit_mean = logit_sum / max(1, sample_count)
    logit_var = max(0.0, logit_sq_sum / max(1, sample_count) - logit_mean ** 2)

    return {
        "loss": running_loss / sample_count,
        "loss_cls": running_cls / sample_count,
        "loss_struct": running_struct / sample_count,
        "loss_neg": running_neg / sample_count,
        "loss_budget": running_budget / sample_count,
        "loss_tv": running_tv / sample_count,
        "loss_comp": running_comp / sample_count,
        "lambda_struct": lambda_struct,
        "logit_mean": logit_mean,
        "logit_std": math.sqrt(logit_var),
        "prob_mean": prob_sum / max(1, sample_count),
        **stats,
    }


def evaluate(net, data_loader, loss_fn, device, tqdm_able):
    net.eval()
    sample_count = 0
    running_loss = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0
    logit_sum = 0.0
    logit_sq_sum = 0.0
    prob_sum = 0.0

    with torch.no_grad():
        with tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
            unit="batch",
            disable=not tqdm_able,
        ) as pbar:
            for x, y, mask in pbar:
                x = x.to(device)
                y = y.to(device).unsqueeze(1).float()
                mask = mask.to(device)

                logits = net(x, padding_mask=mask)
                loss = loss_fn(logits, y)

                sample_count += x.size(0)
                running_loss += loss.item() * x.size(0)

                pred = (logits > 0.0).int()
                y_int = y.int()

                TP += torch.sum((pred == 1) & (y_int == 1)).item()
                FP += torch.sum((pred == 1) & (y_int == 0)).item()
                TN += torch.sum((pred == 0) & (y_int == 0)).item()
                FN += torch.sum((pred == 0) & (y_int == 1)).item()

                logit_sum += logits.detach().sum().item()
                logit_sq_sum += (logits.detach() ** 2).sum().item()
                prob_sum += torch.sigmoid(logits.detach()).sum().item()

                live_stats = summarize_binary_counts(TP, FP, TN, FN)
                pbar.set_postfix(
                    {
                        "loss": running_loss / sample_count,
                        "bal_acc": live_stats["balanced_acc"],
                        "pred_pos": live_stats["pred_pos_rate"],
                        "f1": live_stats["f1"],
                    }
                )

    stats = summarize_binary_counts(TP, FP, TN, FN)
    logit_mean = logit_sum / max(1, sample_count)
    logit_var = max(0.0, logit_sq_sum / max(1, sample_count) - logit_mean ** 2)

    return {
        "loss": running_loss / sample_count,
        "logit_mean": logit_mean,
        "logit_std": math.sqrt(logit_var),
        "prob_mean": prob_sum / max(1, sample_count),
        **stats,
    }


def main():
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    train_cfg = get_dataset_train_cfg(args)

    args.epochs = int(train_cfg.get("epochs", args.epochs))
    args.learning_rate = float(train_cfg.get("learning_rate", args.learning_rate))
    args.batch_size = int(args.batch_size)

    last_best_ckpt_path = None

    for i_iter in range(1):
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_loss_cls": [],
            "train_loss_struct": [],
            "train_loss_neg": [],
            "train_loss_budget": [],
            "train_loss_tv": [],
            "train_loss_comp": [],
            "train_lambda_struct": [],
            "train_precision": [],
            "train_recall": [],
            "train_specificity": [],
            "train_balanced_acc": [],
            "train_f1": [],
            "train_pred_pos_rate": [],
            "train_logit_mean": [],
            "train_logit_std": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_specificity": [],
            "val_balanced_acc": [],
            "val_f1": [],
            "val_pred_pos_rate": [],
            "val_logit_mean": [],
            "val_logit_std": [],
        }

        if args.if_wandb:
            wandb_run_name = f"{args.model}-{args.dataset}-{args.train_gender}-{args.test_gender}"
            wandb.init(project="mamnba_ad", config=args, name=wandb_run_name)
            args = wandb.config

        print(args)

        run_dir = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}"
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(f"{run_dir}/samples", exist_ok=True)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

        if args.model == "DepMamba":
            student_cfg = get_model_cfg(args)
            student_cfg = sanitize_depmamba_cfg(student_cfg)
            net = DepMamba(**student_cfg)
        else:
            raise NotImplementedError(
                f"The {args.model} method has not been implemented by this repo"
            )

        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        train_loader, val_loader, test_loader = build_dataloaders(args)

        if bool(train_cfg.get("debug_overfit_small", False)):
            overfit_pos = int(train_cfg.get("overfit_pos", 8))
            overfit_neg = int(train_cfg.get("overfit_neg", 8))
            train_loader = make_balanced_subset_loader(train_loader, pos_count=overfit_pos, neg_count=overfit_neg, shuffle=True)
            val_loader = make_balanced_subset_loader(train_loader, pos_count=overfit_pos, neg_count=overfit_neg, shuffle=False)
            test_loader = val_loader
            print(f"[DEBUG] balanced overfit subset enabled: pos={overfit_pos}, neg={overfit_neg}, total={overfit_pos + overfit_neg}")

        base_pos = evaluate_constant_baseline(val_loader, predict_positive=True)
        base_neg = evaluate_constant_baseline(val_loader, predict_positive=False)
        print("[Val baseline | all positive]", base_pos)
        print("[Val baseline | all negative]", base_neg)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        weight_decay = float(train_cfg.get("weight_decay", 0.0))

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
            weight_decay=weight_decay,
        )

        struct_cfg = {
            "lambda_struct_peak": float(train_cfg.get("lambda_struct_peak", 0.3)),
            "budget_rho": float(train_cfg.get("budget_rho", 0.25)),
            "comp_margin": float(train_cfg.get("comp_margin", 0.15)),
            "comp_topk_ratio": float(train_cfg.get("comp_topk_ratio", 0.1)),
            "lambda_neg": float(train_cfg.get("lambda_neg", 1.0)),
            "lambda_budget": float(train_cfg.get("lambda_budget", 1.0)),
            "lambda_tv": float(train_cfg.get("lambda_tv", 0.2)),
            "lambda_comp": float(train_cfg.get("lambda_comp", 0.5)),
        }

        best_val_metric = -1.0

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
                    struct_cfg,
                )
                val_results = evaluate(net, val_loader, loss_fn, args.device[0], args.tqdm_able)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_results['loss']:.4f}, "
                    f"train_acc={train_results['acc']:.4f}, "
                    f"train_bal_acc={train_results['balanced_acc']:.4f}, "
                    f"train_pred_pos={train_results['pred_pos_rate']:.4f}, "
                    f"train_cls={train_results['loss_cls']:.4f}, "
                    f"train_struct={train_results['loss_struct']:.4f}, "
                    f"val_loss={val_results['loss']:.4f}, "
                    f"val_acc={val_results['acc']:.4f}, "
                    f"val_bal_acc={val_results['balanced_acc']:.4f}, "
                    f"val_f1={val_results['f1']:.4f}, "
                    f"val_pred_pos={val_results['pred_pos_rate']:.4f}, "
                    f"TP={val_results['TP']}, FP={val_results['FP']}, TN={val_results['TN']}, FN={val_results['FN']}"
                )

                history["train_loss"].append(float(train_results["loss"]))
                history["train_acc"].append(float(train_results["acc"]))
                history["train_loss_cls"].append(float(train_results["loss_cls"]))
                history["train_loss_struct"].append(float(train_results["loss_struct"]))
                history["train_loss_neg"].append(float(train_results["loss_neg"]))
                history["train_loss_budget"].append(float(train_results["loss_budget"]))
                history["train_loss_tv"].append(float(train_results["loss_tv"]))
                history["train_loss_comp"].append(float(train_results["loss_comp"]))
                history["train_lambda_struct"].append(float(train_results["lambda_struct"]))
                history["train_precision"].append(float(train_results["precision"]))
                history["train_recall"].append(float(train_results["recall"]))
                history["train_specificity"].append(float(train_results["specificity"]))
                history["train_balanced_acc"].append(float(train_results["balanced_acc"]))
                history["train_f1"].append(float(train_results["f1"]))
                history["train_pred_pos_rate"].append(float(train_results["pred_pos_rate"]))
                history["train_logit_mean"].append(float(train_results["logit_mean"]))
                history["train_logit_std"].append(float(train_results["logit_std"]))

                history["val_loss"].append(float(val_results["loss"]))
                history["val_acc"].append(float(val_results["acc"]))
                history["val_precision"].append(float(val_results["precision"]))
                history["val_recall"].append(float(val_results["recall"]))
                history["val_specificity"].append(float(val_results["specificity"]))
                history["val_balanced_acc"].append(float(val_results["balanced_acc"]))
                history["val_f1"].append(float(val_results["f1"]))
                history["val_pred_pos_rate"].append(float(val_results["pred_pos_rate"]))
                history["val_logit_mean"].append(float(val_results["logit_mean"]))
                history["val_logit_std"].append(float(val_results["logit_std"]))

                val_metric = val_results["balanced_acc"]

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_ckpt_path = f"{run_dir}/checkpoints/best_model.pt"
                    torch.save(get_core_model(net).state_dict(), best_ckpt_path)
                    last_best_ckpt_path = best_ckpt_path
                    print(
                        f"[Best updated] epoch={epoch:03d}, "
                        f"val_bal_acc={val_results['balanced_acc']:.4f}, "
                        f"val_f1={val_results['f1']:.4f}"
                    )

                if args.if_wandb:
                    wandb.log(
                        {
                            "loss/train": train_results["loss"],
                            "acc/train": train_results["acc"],
                            "loss/train_cls": train_results["loss_cls"],
                            "loss/train_struct": train_results["loss_struct"],
                            "loss/train_neg": train_results["loss_neg"],
                            "loss/train_budget": train_results["loss_budget"],
                            "loss/train_tv": train_results["loss_tv"],
                            "loss/train_comp": train_results["loss_comp"],
                            "lambda/train_struct": train_results["lambda_struct"],
                            "precision/train": train_results["precision"],
                            "recall/train": train_results["recall"],
                            "specificity/train": train_results["specificity"],
                            "balanced_acc/train": train_results["balanced_acc"],
                            "pred_pos_rate/train": train_results["pred_pos_rate"],
                            "logit_mean/train": train_results["logit_mean"],
                            "logit_std/train": train_results["logit_std"],
                            "loss/val": val_results["loss"],
                            "acc/val": val_results["acc"],
                            "precision/val": val_results["precision"],
                            "recall/val": val_results["recall"],
                            "specificity/val": val_results["specificity"],
                            "balanced_acc/val": val_results["balanced_acc"],
                            "f1/val": val_results["f1"],
                            "pred_pos_rate/val": val_results["pred_pos_rate"],
                            "logit_mean/val": val_results["logit_mean"],
                            "logit_std/val": val_results["logit_std"],
                        }
                    )

        best_ckpt_path = f"{run_dir}/checkpoints/best_model.pt"
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

        core_net = get_core_model(net)
        core_net.load_state_dict(torch.load(best_ckpt_path, map_location=args.device[0]))
        core_net.eval()

        with torch.no_grad():
            test_results = evaluate(net, test_loader, loss_fn, args.device[0], args.tqdm_able)
            print("Test results:")
            print(test_results)

            avg_score = (
                test_results["balanced_acc"]
                + test_results["precision"]
                + test_results["recall"]
                + test_results["f1"]
            ) / 4.0

            results_path = f"./results/{args.dataset}_{args.model}_{str(i_iter)}.txt"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                test_result_str = (
                    f'Accuracy:{test_results["acc"]}, '
                    f'BalancedAcc:{test_results["balanced_acc"]}, '
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
    setup_seed(3333)
    main()

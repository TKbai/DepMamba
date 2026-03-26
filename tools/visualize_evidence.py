import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import yaml
import math
import inspect
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from models import DepMamba
from datasets import (
    get_dvlog_dataloader,
    get_lmvd_dataloader,
    get_search_dataloader,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1"):
        return True
    if v in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def sanitize_depmamba_cfg(cfg: dict):
    cfg = dict(cfg)
    if "mm_output_sizes" in cfg and "uni_output_sizes" not in cfg:
        cfg["uni_output_sizes"] = cfg["mm_output_sizes"]

    valid_params = set(inspect.signature(DepMamba.__init__).parameters.keys())
    valid_params.discard("self")
    cfg = {k: v for k, v in cfg.items() if k in valid_params}
    return cfg


def get_model_cfg(full_cfg, dataset):
    if dataset == "dvlog":
        cfg = dict(full_cfg["mmmamba"])
    elif dataset == "lmvd":
        cfg = dict(full_cfg["mmmamba_lmvd"])
    elif dataset == "search":
        cfg = dict(full_cfg["mmmamba_search"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return sanitize_depmamba_cfg(cfg)


def build_loader(full_cfg, dataset, split, batch_size=1):
    data_dir = os.path.join(full_cfg["data_dir"], dataset)
    train_gender = full_cfg.get("train_gender", "both")
    test_gender = full_cfg.get("test_gender", "both")

    if dataset == "dvlog":
        return get_dvlog_dataloader(
            data_dir,
            split,
            batch_size=batch_size,
            gender=test_gender if split != "train" else train_gender,
            aug=False,
        )
    elif dataset == "lmvd":
        return get_lmvd_dataloader(
            data_dir,
            split,
            batch_size=batch_size,
            gender=test_gender if split != "train" else train_gender,
            aug=False,
        )
    elif dataset == "search":
        return get_search_dataloader(
            data_dir,
            split,
            batch_size=batch_size,
            gender=test_gender if split != "train" else train_gender,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


@torch.no_grad()
def collect_probs(model, loader, device):
    probs = []
    labels = []
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        info = model(x, mask, return_intermediate=True)
        prob = torch.sigmoid(info["logits"]).squeeze(1)

        probs.append(prob.detach().cpu())
        labels.append(y.detach().cpu())

    probs = torch.cat(probs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy().astype(int)
    return probs, labels


def calc_metrics_from_probs(probs, labels, threshold):
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    TP = int(((preds == 1) & (labels == 1)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    acc = (TP + TN) / max(1, len(labels))
    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    specificity = TN / max(1, TN + FP)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    balanced_acc = 0.5 * (recall + specificity)
    pred_pos_rate = preds.mean() if len(preds) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "pred_pos_rate": float(pred_pos_rate),
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


def sweep_best_threshold(model, val_loader, device):
    probs, labels = collect_probs(model, val_loader, device)

    best = None
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        m = calc_metrics_from_probs(probs, labels, thr)
        score = (m["balanced_acc"], m["f1"])
        if best is None or score > best:
            best = score
            best_thr = float(thr)

    metrics = calc_metrics_from_probs(probs, labels, best_thr)
    return best_thr, metrics


@torch.no_grad()
def pick_cases(model, loader, device, threshold):
    """
    自动挑 4 个样本：
    TP / TN / FP / FN 各一个
    返回: dict(kind -> dataset_index)
    """
    picked = {}

    for idx, (x, y, mask) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        info = model(x, mask, return_intermediate=True)
        prob = torch.sigmoid(info["logits"]).item()
        pred = int(prob >= threshold)
        true = int(y.item())

        if true == 1 and pred == 1:
            kind = "TP"
        elif true == 0 and pred == 0:
            kind = "TN"
        elif true == 0 and pred == 1:
            kind = "FP"
        else:
            kind = "FN"

        if kind not in picked:
            picked[kind] = idx

        if len(picked) == 4:
            break

    return picked


def parse_indices(s):
    if s is None or s == "":
        return None
    return [int(x) for x in s.split(",")]


def get_case_kind(true, pred):
    if true == 1 and pred == 1:
        return "TP"
    elif true == 0 and pred == 0:
        return "TN"
    elif true == 0 and pred == 1:
        return "FP"
    else:
        return "FN"


def tensor_1d(info, key, length):
    x = info[key][0, :length].detach().cpu().numpy()
    return x


def shade_spans(ax, spans, color="gray", alpha=0.12):
    if spans is None:
        return
    for sp in spans:
        ax.axvspan(sp["start"], sp["end"], color=color, alpha=alpha)


def plot_one_case(info, y_true, prob, threshold, save_png, save_npz):
    mask = info["audio_proposal"].new_ones(info["audio_proposal"].shape)
    # valid length 优先从 attn 长度推；若模型里已把 pad 清零，这里也可从 non-zero mask 推
    if "audio_attn" in info:
        length = int((info["audio_attn"][0] > -1).sum().item())
    else:
        length = int(info["audio_proposal"].shape[1])

    # 更稳：如果有 padding，全局从非 -1e4 的聚合 logit 推长度
    if "audio_agg_logit" in info:
        valid = (info["audio_agg_logit"][0] > -1e3).detach().cpu().numpy()
        length = int(valid.sum())

    a_p = tensor_1d(info, "audio_proposal", length)
    a_r = tensor_1d(info, "audio_refined", length)
    a_h = tensor_1d(info, "audio_corroborated", length)
    a_attn = tensor_1d(info, "audio_attn", length)

    v_p = tensor_1d(info, "video_proposal", length)
    v_r = tensor_1d(info, "video_refined", length)
    v_h = tensor_1d(info, "video_corroborated", length)
    v_attn = tensor_1d(info, "video_attn", length)

    a_spans = None
    v_spans = None
    if "audio_spans" in info:
        a_spans = info["audio_spans"][0]
    if "video_spans" in info:
        v_spans = info["video_spans"][0]

    pred = int(prob >= threshold)
    kind = get_case_kind(int(y_true), pred)

    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True)

    xs = np.arange(length)

    # audio row
    axes[0, 0].plot(xs, a_p)
    axes[0, 0].set_title("Audio proposal p")
    axes[0, 1].plot(xs, a_r)
    shade_spans(axes[0, 1], a_spans)
    axes[0, 1].set_title("Audio refined r")
    axes[0, 2].plot(xs, a_h)
    shade_spans(axes[0, 2], a_spans)
    axes[0, 2].set_title("Audio corroborated hat_r")
    axes[0, 3].plot(xs, a_attn)
    axes[0, 3].set_title("Audio aggregation attn")

    # video row
    axes[1, 0].plot(xs, v_p)
    axes[1, 0].set_title("Video proposal p")
    axes[1, 1].plot(xs, v_r)
    shade_spans(axes[1, 1], v_spans)
    axes[1, 1].set_title("Video refined r")
    axes[1, 2].plot(xs, v_h)
    shade_spans(axes[1, 2], v_spans)
    axes[1, 2].set_title("Video corroborated hat_r")
    axes[1, 3].plot(xs, v_attn)
    axes[1, 3].set_title("Video aggregation attn")

    for ax in axes.reshape(-1):
        ax.set_ylim(bottom=min(-0.02, ax.get_ylim()[0]))
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{kind} | y={int(y_true)} | prob={prob:.4f} | thr={threshold:.2f}",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(save_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        save_npz,
        y_true=int(y_true),
        prob=float(prob),
        threshold=float(threshold),
        audio_proposal=a_p,
        audio_refined=a_r,
        audio_corroborated=a_h,
        audio_attn=a_attn,
        video_proposal=v_p,
        video_refined=v_r,
        video_corroborated=v_h,
        video_attn=v_attn,
    )


@torch.no_grad()
def run_and_plot_selected(model, loader, device, threshold, indices, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    selected = set(indices)
    summary = []

    for idx, (x, y, mask) in enumerate(loader):
        if idx not in selected:
            continue

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        info = model(x, mask, return_intermediate=True)
        prob = torch.sigmoid(info["logits"]).item()
        pred = int(prob >= threshold)
        true = int(y.item())
        kind = get_case_kind(true, pred)

        png_path = os.path.join(save_dir, f"idx_{idx:04d}_{kind}.png")
        npz_path = os.path.join(save_dir, f"idx_{idx:04d}_{kind}.npz")
        plot_one_case(info, true, prob, threshold, png_path, npz_path)

        summary.append(
            {
                "index": idx,
                "kind": kind,
                "y_true": true,
                "pred": pred,
                "prob": float(prob),
                "png": png_path,
                "npz": npz_path,
            }
        )

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["dvlog", "lmvd", "search"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--use_corroboration", type=str2bool, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    model_cfg = get_model_cfg(full_cfg, args.dataset)
    if args.use_corroboration is not None:
        model_cfg["use_corroboration"] = args.use_corroboration

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = DepMamba(**model_cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    val_loader = build_loader(full_cfg, args.dataset, "valid", batch_size=1)
    target_loader = build_loader(full_cfg, args.dataset, args.split, batch_size=1)

    if args.threshold is None:
        best_thr, best_val_metrics = sweep_best_threshold(model, val_loader, device)
        print("[best threshold from val]")
        print(best_val_metrics)
    else:
        best_thr = float(args.threshold)
        print(f"[use fixed threshold] {best_thr:.4f}")

    indices = parse_indices(args.indices)
    if indices is None:
        picked = pick_cases(model, target_loader, device, best_thr)
        print("[auto picked cases]")
        print(picked)
        indices = list(picked.values())
    else:
        print("[use given indices]")
        print(indices)

    summary = run_and_plot_selected(
        model=model,
        loader=target_loader,
        device=device,
        threshold=best_thr,
        indices=indices,
        save_dir=args.save_dir,
    )
    print("[saved summary]")
    print(summary)


if __name__ == "__main__":
    main()
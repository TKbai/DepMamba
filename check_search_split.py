import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/home/ac/data/bai/DepMamba-main/datasets/search")
TRAIN_PKL = DATA_ROOT / "SEARCH_data_train.pkl"
TEST_PKL = DATA_ROOT / "SEARCH_data_test.pkl"

VALID_RATIO = 0.2
SPLIT_SEED = 3333
SCORE_THRESHOLD = 10


def bin_label_from_score(score, threshold=10):
    return int(int(score) >= threshold)


def summarize_subset(name, data_list):
    y_bin = [bin_label_from_score(x[3], SCORE_THRESHOLD) for x in data_list]
    y_score = [int(x[3]) for x in data_list]
    y_cls = [int(x[4]) for x in data_list]

    pos = int(np.sum(y_bin))
    neg = len(y_bin) - pos
    pos_ratio = pos / max(1, len(y_bin))

    print(f"\n===== {name} =====")
    print(f"total      : {len(y_bin)}")
    print(f"positive   : {pos}")
    print(f"negative   : {neg}")
    print(f"pos_ratio  : {pos_ratio:.4f}")
    print(f"score_dist : {dict(sorted(Counter(y_score).items()))}")
    print(f"cls_dist   : {dict(sorted(Counter(y_cls).items()))}")


def main():
    with open(TRAIN_PKL, "rb") as f:
        all_train = pickle.load(f)

    with open(TEST_PKL, "rb") as f:
        test_data = pickle.load(f)

    n = len(all_train)
    indices = list(range(n))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices)

    split = int(n * (1.0 - VALID_RATIO))
    train_idx = indices[:split]
    valid_idx = indices[split:]

    train_data = [all_train[i] for i in train_idx]
    valid_data = [all_train[i] for i in valid_idx]

    print("split config:")
    print(f"VALID_RATIO     = {VALID_RATIO}")
    print(f"SPLIT_SEED      = {SPLIT_SEED}")
    print(f"SCORE_THRESHOLD = {SCORE_THRESHOLD}")

    summarize_subset("train", train_data)
    summarize_subset("valid", valid_data)
    summarize_subset("test", test_data)


if __name__ == "__main__":
    main()
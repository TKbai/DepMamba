import pickle
import numpy as np
from collections import Counter

pkl_path = "/home/ac/data/bai/DepMamba-main/datasets/SEARCH/SEARCH_data_train.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("data type:", type(data))
print("data len :", len(data))

sample = data[0]
print("sample type:", type(sample))
print("sample len :", len(sample))

print("\ncheck last two fields:")
print("field -2:", Counter([x[-2] for x in data[:100]]))
print("field -1:", Counter([x[-1] for x in data[:100]]))

for i, v in enumerate(sample):
    print(f"\n--- item {i} ---")
    print("type:", type(v))
    if hasattr(v, "shape"):
        print("shape:", v.shape)
        print("dtype:", v.dtype)
        try:
            print("min/max:", np.min(v), np.max(v))
        except Exception:
            pass
    else:
        print("value:", v)

for idx in range(5):
    s = data[idx]
    print(f"\n===== sample {idx} =====")
    for i, v in enumerate(s):
        if hasattr(v, "shape"):
            print(i, type(v), v.shape, v.dtype)
        else:
            print(i, type(v), v)
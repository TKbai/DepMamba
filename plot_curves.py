import json
import matplotlib.pyplot as plt

path = "/home/ac/data/bai/DepMamba-main/savefile/dvlog_DepMamba_0/curves.json"  # 换成你自己的
with open(path) as f:
    h = json.load(f)

epochs = range(len(h["train_acc"]))   # 0~N-1，没问题

plt.figure(figsize=(15, 4))

# ---- 子图 1：train / val acc ----
plt.subplot(1, 3, 1)
plt.plot(epochs, h["train_acc"], label="train_acc")
plt.plot(epochs, h["val_acc"],   label="val_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

# ---- 子图 2：gate mean ----
plt.subplot(1, 3, 2)
plt.plot(epochs, h["gate_a_mean"], label="audio_gate_mean")
plt.plot(epochs, h["gate_v_mean"], label="video_gate_mean")
plt.xlabel("epoch")
plt.ylabel("gate mean")
plt.title("Gate stats")
plt.legend()
plt.grid(True)

# ---- 子图 3：val precision / recall / f1 ----
plt.subplot(1, 3, 3)
plt.plot(epochs, h["val_precision"], label="val_precision")
#plt.plot(epochs, h["val_recall"],    label="val_recall")
#plt.plot(epochs, h["val_f1"],        label="val_f1")
plt.xlabel("epoch")
plt.ylabel("score")
plt.title("Val P / R / F1")
plt.legend()
plt.grid(True)

plt.tight_layout()

out_path = "curve_result_prf0.png"
plt.savefig(out_path, dpi=200)
print("绘图已保存为", out_path)

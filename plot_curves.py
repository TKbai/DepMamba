import json
import matplotlib.pyplot as plt

path = "/home/ac/data/bai/DepMamba-main/savefile/dvlog_DepMamba_2/curves.json"  # 换成你自己的
with open(path) as f:
    h = json.load(f)

epochs = range(len(h["train_acc"]))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, h["train_acc"], label="train_acc")
plt.plot(epochs, h["val_acc"],   label="val_acc")
plt.legend(); plt.grid(True); plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(epochs, h["gate_a_mean"], label="audio_gate_mean")
plt.plot(epochs, h["gate_v_mean"], label="video_gate_mean")
plt.legend(); plt.grid(True); plt.xlabel("epoch")

plt.tight_layout()
plt.savefig('curve_result2.png') # <--- 换成这一行
print("绘图已保存为 curve_result.png")

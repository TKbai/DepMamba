import torch
from models.evidence_selector import EvidenceSelector

def run_once(tau, hard):
    B, D, L = 2, 128, 1000
    x = torch.randn(B, D, L)

    selector = EvidenceSelector(d_in=D, d_hidden=64, tau=tau, hard=hard)
    gate, logits = selector(x)

    print(f"tau={tau}, hard={hard}")
    print("gate shape:", gate.shape)  # 期望 (2, 1, 1000)
    print("gate stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
        gate.min().item(), gate.max().item(), gate.mean().item()
    ))
    print("-" * 40)

if __name__ == "__main__":
    run_once(tau=5.0, hard=False)   # 高温, 应该接近 0.5 左右
    run_once(tau=0.1, hard=False)   # 低温, 分布更尖锐
    run_once(tau=0.1, hard=True)    # 近似 0/1
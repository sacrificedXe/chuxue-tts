"""训练反演网络：mel(80,T) + f0 → 声道参数(4,T)。

目标不是“猜中真参数”（同一频谱对应多组声道形状，参数不可辨识），
而是让网络学到 VTL 域的合理映射，最终判据放在音频层（LTAS 距离）。

    python train.py --data data --epochs 30 --out model.pt
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_mels=80, hid=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_mels + 1, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 5, padding=4, dilation=2), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 5, padding=8, dilation=4), nn.BatchNorm1d(hid), nn.ReLU(),
        )
        self.rnn = nn.GRU(hid, 96, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 4))

    def forward(self, mel, f0):
        # mel: (B,T,80)  f0: (B,T) Hz
        x = torch.cat([mel, (torch.log(f0.clamp(60, 500) / 120.0))[..., None]], -1)
        x = x.transpose(1, 2)
        x = self.enc(x).transpose(1, 2)
        x, _ = self.rnn(x)
        return torch.sigmoid(self.head(x))            # (B,T,4) in [0,1]


def load(data):
    mels, pars, f0s = [], [], []
    for f in sorted(glob.glob(os.path.join(data, 'shard_*.npz'))):
        z = np.load(f)
        mels.append(z['mel'].astype(np.float32))
        pars.append(z['param'].astype(np.float32))
        f0s.append(z['f0'].astype(np.float32))
    return np.concatenate(mels), np.concatenate(pars), np.concatenate(f0s)


def augment(xb):
    """训练时加噪声与频谱倾斜，模拟真人录音的毛刺与染色，缩小域偏移。"""
    b, t, d = xb.shape
    noise = torch.randn_like(xb) * 0.12
    ramp = torch.linspace(-1, 1, d)
    tilt = (torch.rand(b, t, 1) * 0.25 - 0.125) * ramp
    y = xb + noise + tilt
    y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-6)
    return y.clamp(-6, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='model.pt')
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    mel, par, f0 = load(a.data)
    n = len(mel)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nval = min(max(8, int(n * 0.05)), max(1, n // 5))
    val, tr = idx[:nval], idx[nval:]
    print('samples', n, 'train', len(tr), 'val', len(val), flush=True)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = 9e9

    tm = torch.from_numpy(mel).float()
    tp = torch.from_numpy(par).float()
    tf = torch.from_numpy(f0).float()

    def run(batch_idx, train=True):
        loss_sum, mae_sum, nb = 0.0, 0.0, 0
        order = np.array(batch_idx)
        if train:
            order = rng.permutation(order)
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            if len(b) < 2:
                continue
            xb, pb, fb = tm[b], tp[b], tf[b]
            if train:
                xb = augment(xb)
            pred = net(xb, fb)
            loss = nn.functional.mse_loss(pred, pb)
            # 时间平滑：参数轨迹二阶差分惩罚，抑制逐帧抖动
            if pred.shape[1] > 2:
                loss = loss + 0.05 * nn.functional.mse_loss(
                    pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2], torch.zeros_like(pred[:, 2:]))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * len(b)
            mae_sum += float((pred - pb).abs().mean()) * len(b)
            nb += len(b)
        return loss_sum / max(1, nb), mae_sum / max(1, nb)

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); ltr, mtr = run(tr, True)
        net.eval()
        with torch.no_grad():
            lva, mva = run(val, False)
        sched.step()
        print('ep %02d  train mse %.5f mae %.3f | val mse %.5f mae %.3f | %.0fs'
              % (ep, ltr, mtr, lva, mva, time.time() - t0), flush=True)
        if lva < best:
            best = lva
            torch.save({'state': net.state_dict(), 'val_mse': lva, 'val_mae': mva,
                        'epoch': ep, 'n': n}, a.out)
    json.dump(dict(best_val_mse=best, out=a.out, samples=n), open('train_summary.json', 'w'), indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()

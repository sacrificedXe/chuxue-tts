"""推理 + 评估：真人语音 → mel → 参数 → VTL 合成 → 与原音频比 LTAS。

    python infer.py --wav cvcn.wav --model model.pt --out vtl_out.wav --max-sec 60

输出 json：LTAS 距离（本项目主判据）、中性参数基线、参数范围（判断是否塌缩）。
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import make_data as MD
import vtl_lib as V
import torch

from train import Net


def norm_mel(m):
    return m - m.mean(-1, keepdims=True)


def read_audio(path, max_sec=None):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    if sr != V.SR:
        g = np.gcd(int(sr), V.SR)
        x = resample_poly(x, V.SR // g, int(sr) // g).astype(np.float32)
        sr = V.SR
    if max_sec:
        x = x[: int(max_sec * sr)]
    x = x / max(1e-6, np.abs(x).max()) * 0.9
    return x, sr


def f0_track(x, sr=V.SR):
    """逐模型帧估 F0（自相关），返回 (T,) 与浊音掩码 (T,)。"""
    win = V.MODEL_HOP * V.BLOCK
    T = len(x) // win
    f0 = np.zeros(T, np.float32)
    voiced = np.zeros(T, bool)
    last = 150.0
    for i in range(T):
        seg = x[i * win:(i + 1) * win]
        e = float(np.sqrt((seg ** 2).mean()))
        s = seg - seg.mean()
        c = np.correlate(s, s, 'full')[len(s) - 1:]
        c = c / (c[0] + 1e-12)
        lo, hi = int(sr / 450), int(sr / 60)
        if e < 0.01 or hi >= len(c):
            f0[i] = last
            continue
        k = int(np.argmax(c[lo:hi])) + lo
        if c[k] > 0.35:
            last = float(sr / k)
            voiced[i] = True
        f0[i] = last
    return f0, voiced


def ltas(x, sr=V.SR, grid=None):
    if grid is None:
        grid = np.geomspace(200, 5000, 41)
    x = x.astype(np.float64)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    v = []
    for a, b in zip(grid[:-1], grid[1:]):
        sel = (f >= a) & (f < b)
        v.append(np.sqrt((S[sel] ** 2).mean()) if sel.sum() else 1e-9)
    v = 20 * np.log10(np.array(v) + 1e-12)
    return v - v.mean()


def ltas_dist(a, b):
    return float(np.sqrt(((ltas(a) - ltas(b)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--model', default='model.pt')
    ap.add_argument('--out', default='vtl_out.wav')
    ap.add_argument('--max-sec', type=float, default=60.0)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    x, sr = read_audio(a.wav, a.max_sec)
    mel = MD.make_mel(x).astype(np.float32)          # (T,80)
    f0, voiced = f0_track(x)
    T = min(len(mel), len(f0))
    mel, f0 = mel[:T], f0[:T]
    print('frames', T, 'voiced %.1f%%' % (100 * voiced[:T].mean()), flush=True)

    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    net = Net(); net.load_state_dict(ck['state']); net.eval()
    xs = torch.from_numpy(norm_mel(mel)).float()[None]
    fs = torch.from_numpy(f0).float()[None]
    with torch.no_grad():
        pred = net(xs, fs)[0].numpy()                 # (T,4) unit
    params = V.from_unit(pred)

    vtl = V.VTL()
    y = vtl.synth(params, f0)
    y = y / max(1e-6, np.abs(y).max()) * 0.9
    n = min(len(x), len(y))
    sf.write(a.out, y, V.SR)

    d = ltas_dist(y[:n], x[:n])
    # 基线：中性参数，同样时长与 F0
    y0 = vtl.synth(np.tile(V.NEUTRAL, (T, 1)), f0)
    y0 = y0 / max(1e-6, np.abs(y0).max()) * 0.9
    n0 = min(len(x), len(y0))
    d0 = ltas_dist(y0[:n0], x[:n0])

    sp = 100 * params.std(0) / (V.HI - V.LO)
    out = dict(frames=T, ltas_model=round(d, 2), ltas_neutral=round(d0, 2),
               param_std_pct=[round(float(v), 1) for v in sp],
               param_mean=[round(float(v), 2) for v in params.mean(0)],
               param_train_mean_mae=ck.get('val_mae'),
               seconds=round(len(y) / V.SR, 1))
    print(json.dumps(out, ensure_ascii=False))
    json.dump(out, open('infer_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

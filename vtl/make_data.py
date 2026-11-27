"""生成 VTL 合成数据集：随机声道轨迹 → 音频 → mel + 参数标签。

    python make_data.py --n 6000 --out data --shard 500 --workers 4

每个样本 12 个模型帧（≈0.55 s）。输出 data/shard_XXX.npz：
    mel   (T, 80) float16   对数 mel（0.55 s 内时间平均到 46 ms/帧）
    param (T, 4)  float16   归一化到 [0,1] 的 TCX/TCY/JA/LD
    f0    (T,)    float32
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import vtl_lib as V

N_FFT = 1024
N_MELS = 80
FMIN, FMAX = 50.0, 8000.0


def _mel_bank(sr=V.SR, n_fft=N_FFT, n_mels=N_MELS):
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    m = np.linspace(hz2mel(FMIN), hz2mel(FMAX), n_mels + 2)
    hz = mel2hz(m)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo + 1e-9),
                                   (hi - freqs) / (hi - c + 1e-9)), 0, None)
    return fb


_BANK = {}


def make_mel(audio, sr=V.SR):
    if 'fb' not in _BANK:
        _BANK['fb'] = _mel_bank(sr)
    win = np.hanning(N_FFT).astype(np.float32)
    T = len(audio) // (V.MODEL_HOP * V.BLOCK)
    n = T * V.MODEL_HOP
    need = (n - 1) * V.BLOCK + N_FFT
    if len(audio) < need:
        audio = np.concatenate([audio, np.zeros(need - len(audio), dtype=audio.dtype)])
    idx = np.arange(N_FFT)[None] + V.BLOCK * np.arange(n)[:, None]
    frames = audio[idx] * win
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2           # (n, 513)
    mel = S @ _BANK['fb'].T                                # (n, 80)
    mel = mel[:T * V.MODEL_HOP].reshape(T, V.MODEL_HOP, N_MELS).mean(1)   # (T, 80)
    mel = np.log(mel + 1e-6)
    # 逐帧标准化：同步去掉音量与频谱对比度。真人录音与 VTL 合成在这两项上差 3 倍，
    # 不归一化时网络会把真实频谱当成分布外输入，直接塌成常数输出。
    mel = (mel - mel.mean(-1, keepdims=True)) / (mel.std(-1, keepdims=True) + 1e-6)
    return np.clip(mel, -6, 6)


_G = {}


def _init():
    _G['vtl'] = V.VTL()


def gen_one(seed):
    rng = np.random.default_rng(seed)
    T = 12
    u0 = rng.uniform(0.12, 0.88, 4)
    u1 = np.clip(u0 + rng.normal(0, 0.22, 4), 0.08, 0.92)
    t = np.linspace(0, 1, T)[:, None]
    u = u0 * (1 - t) + u1 * t
    u += rng.normal(0, 0.012, (T, 4))
    u = np.clip(u, 0.0, 1.0)
    f0b = rng.uniform(85, 320)
    f0 = f0b * (1 + 0.04 * np.sin(np.linspace(0, rng.uniform(1, 4), T) + rng.uniform(0, 6)))
    f0 += rng.normal(0, 1.5, T)
    audio = _G['vtl'].synth(V.from_unit(u), f0)
    if np.abs(audio).max() < 1e-4:
        return None
    mel = make_mel(audio)
    return mel.astype(np.float16), u.astype(np.float16), f0.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--out', default='data')
    ap.add_argument('--shard', type=int, default=500)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    buf, shard, total = [], 0, 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(gen_one, range(a.n), chunksize=8)):
            if r is None:
                continue
            buf.append(r)
            total += 1
            if len(buf) >= a.shard:
                _save(buf, a.out, shard); shard += 1; buf = []
            if k % 200 == 0:
                el = time.time() - t0
                print('%d/%d  %.0fs  (%.2f s/sample)' % (k, a.n, el, el / max(1, k)), flush=True)
    if buf:
        _save(buf, a.out, shard); shard += 1
    meta = dict(n=total, shards=shard, seconds=time.time() - t0,
                frames_per_sample=12, model_hop=V.MODEL_HOP, sr=V.SR, free=V.FREE)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print('done', meta)


def _save(buf, out, idx):
    mel = np.stack([b[0] for b in buf])
    par = np.stack([b[1] for b in buf])
    f0 = np.stack([b[2] for b in buf])
    np.savez_compressed(os.path.join(out, 'shard_%03d.npz' % idx), mel=mel, param=par, f0=f0)


if __name__ == '__main__':
    main()

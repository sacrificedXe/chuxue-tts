"""统一的特征提取。训练和推理必须调用同一个函数，否则会重现"归一化不一致"的错。

输入：原始对数 mel（..., T, 80，未做任何归一化）
输出：(..., T, 161)
  0:80    mel 去帧均值（保留频谱倾斜 = 亮度信息）
  80:160  帧间差分（保留"在怎么动"）
  160     该帧的对比度（标准差，取对数）
"""
import numpy as np

N_FEAT = 161


def feat(mel_raw):
    m = np.asarray(mel_raw, dtype=np.float32)
    mc = m - m.mean(-1, keepdims=True)                  # 只去音量，不去倾斜
    d = np.diff(mc, axis=-2, prepend=mc[..., :1, :])     # 帧间差分
    s = np.log(m.std(-1)[..., None] + 1e-6)             # 对比度
    return np.concatenate([mc, d, s], -1).astype(np.float32)

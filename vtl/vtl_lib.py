"""VTL 公共库：ctypes 封装 + 特征提取。Actions 与本机共用。"""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get('VTL_LIB', os.path.join(HERE, 'libVTL_anat.so'))
SPK = os.environ.get('VTL_SPK', os.path.join(HERE, 'W02.speaker'))
SR = 48000
BLOCK = 110                      # 一次 VTL 调用合成的采样数 ≈ 2.29 ms
MODEL_HOP = 20                   # 模型帧 = 20 个 BLOCK ≈ 45.8 ms（≈22 fps）

PARAM_NAMES = ['HX', 'HY', 'JX', 'JA', 'LP', 'LD', 'VS', 'VO',
               'TCX', 'TCY', 'TTX', 'TTY', 'TBX', 'TBY', 'TRX', 'TRY',
               'TS1', 'TS2', 'TS3']
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
BASE = np.array([0.17, -3.94, 0.0, -4.15, 0.07, 0.99, 0.8, -0.1,
                 0.15, -1.83, 4.25, -1.69, 2.55, -0.68, -2.84, -2.9,
                 0.21, 0.04, 0.15])
FREE = ['TCX', 'TCY', 'JA', 'LD']
RANGE = {'TCX': (-3.0, 4.0), 'TCY': (-3.0, 1.0), 'JA': (-7.0, 0.0), 'LD': (-2.0, 2.0)}
NEUTRAL = np.array([0.15, -1.83, -4.15, 0.99])
LO = np.array([RANGE[n][0] for n in FREE])
HI = np.array([RANGE[n][1] for n in FREE])


def to_unit(p):
    return (np.asarray(p) - LO) / (HI - LO)


def from_unit(u):
    return LO + np.clip(np.asarray(u), 0, 1) * (HI - LO)


class VTL:
    def __init__(self, spk=SPK, libpath=LIB):
        lib = ctypes.CDLL(libpath)
        f64p = np.ctypeslib.ndpointer(np.float64, flags='C')
        lib.vtlInitialize.argtypes = [ctypes.c_char_p]
        lib.vtlInitialize.restype = ctypes.c_int
        lib.vtlGetConstants.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
        lib.vtlResetTractSynthesis.argtypes = []
        lib.vtlResetTractSynthesis.restype = ctypes.c_int
        lib.vtlSynthesisAddTract.argtypes = [ctypes.c_int, f64p, f64p, f64p]
        lib.vtlSynthesisAddTract.restype = ctypes.c_int
        rc = lib.vtlInitialize(spk.encode())
        if rc != 0:
            raise RuntimeError('vtlInitialize rc=%d' % rc)
        a = ctypes.c_int(); nt = ctypes.c_int(); ntr = ctypes.c_int(); ng = ctypes.c_int()
        lib.vtlGetConstants(ctypes.byref(a), ctypes.byref(nt), ctypes.byref(ntr), ctypes.byref(ng))
        self.lib = lib
        self.tract_dim = nt.value
        self.glottis_dim = ng.value
        self.buf = np.zeros(4000)
        self.gl = np.zeros(ng.value)
        self.gl[1] = 8000.0
        self.gl[4] = 0.03
        self.reset()

    def reset(self):
        self.lib.vtlResetTractSynthesis()
        self.lib.vtlSynthesisAddTract(0, self.buf, self.tract(), self.gl)   # 首帧只用于初始化

    def tract(self, p=None):
        t = BASE.copy()
        if p is not None:
            for name, v in zip(FREE, np.atleast_1d(p)):
                t[IDX[name]] = float(v)
        return t

    def synth(self, params, f0):
        """params: (T,4) 实际单位；f0: (T,) Hz。返回长度 T*BLOCK 的音频。"""
        params = np.atleast_2d(params)
        f0 = np.atleast_1d(f0).astype(float)
        T = params.shape[0]
        # 逐 BLOCK 线性插值，避免帧边界跳变引起的咔哒声
        n = T * MODEL_HOP
        idx = np.linspace(0, T - 1, n)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T - 1)
        w = (idx - lo)[:, None]
        p_block = params[lo] * (1 - w) + params[hi] * w
        f_block = np.clip(f0[lo] * (1 - w[:, 0]) + f0[hi] * w[:, 0], 60.0, 500.0)
        self.reset()
        chunks = []
        for i in range(n):
            tract = self.tract(p_block[i])
            self.gl[0] = float(f_block[i])
            self.buf[:] = 0
            self.lib.vtlSynthesisAddTract(BLOCK, self.buf, tract, self.gl)
            chunks.append(self.buf[:BLOCK].copy())
        return np.concatenate(chunks)


def parse_speaker_f0(spk=SPK):
    for line in open(spk, encoding='utf-8', errors='ignore'):
        if line.lower().startswith('f0'):
            try:
                return float(line.split(':')[-1].split()[0])
            except Exception:
                pass
    return 120.0

"""VTL 公共库：ctypes 封装 + 特征提取。Actions 与本机共用。"""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve():
    """优先用仓库内自带的库和音色；没有则回落到 pip 包 vocaltractlab。"""
    lib = os.environ.get('VTL_LIB')
    spk = os.environ.get('VTL_SPK')
    if not lib:
        cand = os.path.join(HERE, 'libVTL_anat.so')
        lib = cand if os.path.exists(cand) else None
    if not spk:
        cand = os.path.join(HERE, 'W02.speaker')
        spk = cand if os.path.exists(cand) else None
    if lib is None or spk is None:
        try:
            import vocaltractlab_cython as vc
            base = os.path.dirname(vc.__file__)
            lib = lib or os.path.join(base, 'libVocalTractLabApi.so')
            spk = spk or os.path.join(base, 'resources', 'JD3.speaker')
        except Exception as e:
            raise RuntimeError('找不到 VTL 库/音色：%s' % e)
    return lib, spk


LIB, SPK = _resolve()
SR = 48000
BLOCK = 110                      # 一次 VTL 调用合成的采样数 ≈ 2.29 ms
MODEL_HOP = 20                   # 模型帧 = 20 个 BLOCK ≈ 45.8 ms（≈22 fps）

GLOTTIS_DEFAULT = np.array([120.0, 8000.0, 0.02, 0.02, 0.03, 1.22, 1.0, 0.0, 25.0])

PARAM_NAMES = ['HX', 'HY', 'JX', 'JA', 'LP', 'LD', 'VS', 'VO',
               'TCX', 'TCY', 'TTX', 'TTY', 'TBX', 'TBY', 'TRX', 'TRY',
               'TS1', 'TS2', 'TS3']
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
BASE = np.array([0.17, -3.94, 0.0, -4.15, 0.07, 0.99, 0.8, -0.1,
                 0.15, -1.83, 4.25, -1.69, 2.55, -0.68, -2.84, -2.9,
                 0.21, 0.04, 0.15])
FREE = ['TCX', 'TCY', 'JA', 'LD']
# 实测可发声域（超出即闭管静音）：LD≤0 闭唇、TCY≥-0.6 舌顶腭、TCX≥2.6 或 ≤-1.6 闭管
RANGE = {'TCX': (-0.6, 2.4), 'TCY': (-3.0, -0.9), 'JA': (-6.5, -0.5), 'LD': (0.3, 1.8)}
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
        self.tract_buf = BASE.copy()   # 必须复用同一块内存：VTL 会保留上一次的形状指针做插值
        self.gl = np.zeros(ng.value)
        self.gl[:] = GLOTTIS_DEFAULT
        self.reset()

    def reset(self):
        self.tract_buf[:] = BASE
        self.lib.vtlResetTractSynthesis()
        self.lib.vtlSynthesisAddTract(0, self.buf, self.tract_buf, self.gl)   # 首帧只用于初始化

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
            self.tract_buf[:] = BASE
            for name, v in zip(FREE, p_block[i]):
                self.tract_buf[IDX[name]] = float(v)
            self.gl[0] = float(f_block[i])
            self.buf[:] = 0
            self.lib.vtlSynthesisAddTract(BLOCK, self.buf, self.tract_buf, self.gl)
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

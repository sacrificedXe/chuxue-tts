import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gsv')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'GPT_SoVITS'))
os.chdir(ROOT)
torch.set_num_threads(4)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

here = os.path.dirname(os.path.abspath(__file__))
ckpt = glob.glob(os.path.join(here, 'cx', '**', '*.ckpt'), recursive=True)[0]
pth = glob.glob(os.path.join(here, 'cx', '**', '*.pth'), recursive=True)[0]
_soft = os.path.join(here, 'ref_soft.wav')
ref = _soft if os.path.exists(_soft) else glob.glob(
    os.path.join(here, 'cx', '**', '*reference_audios*', '**', '*.wav'), recursive=True)[0]
prompt_text = ('出来玩耍就要尽兴，别把那些条条框框套在我身上呀。' if os.path.exists(_soft)
               else '啊，我是现任剩女，没错。')

cfg = TTS_Config('GPT_SoVITS/configs/tts_infer.yaml')
cfg.device = 'cpu'
cfg.is_half = False
cfg.version = 'v4'
cfg.bert_base_path = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
cfg.cnhuhbert_base_path = 'GPT_SoVITS/pretrained_models/chinese-hubert-base'
cfg.t2s_weights_path = ckpt
cfg.vits_weights_path = pth

tts = TTS(cfg)
print('models loaded', flush=True)

text = os.environ.get('TARGET_TEXT') or '主啊，是时候了|夏天盛极一时|把你的阴影置于日晷上|让风吹过牧场'
segments = [s.strip() for s in text.split('|') if s.strip()]
print('segments:', len(segments), flush=True)


def envelope(x, sr, win=0.02, smooth=0.06):
    w = int(win * sr)
    e = np.sqrt(np.convolve(np.pad(x ** 2, (w // 2, w // 2), mode='edge'),
                            np.ones(w) / w, 'valid'))[:len(x)]
    g = int(smooth * sr)
    e = np.convolve(np.pad(e, (g // 2, g // 2), mode='edge'), np.ones(g) / g, 'valid')[:len(x)]
    return e / (e.max() + 1e-9)


def breath_layer(y, sr, rng):
    """三层气音：气声贴合浊音、句首吸气声、句尾气散。"""
    env = envelope(y, sr)

    # 1) 气声：400-2500 Hz，只跟浊音包络
    n1 = rng.standard_normal(len(y))
    n1 = sosfilt(butter(2, [400, 2500], 'bp', fs=sr, output='sos'), n1)
    n1 = n1 / (np.sqrt((n1 ** 2).mean()) + 1e-12)
    y = y + n1 * env * 0.07 * np.abs(y).max()

    # 2) 句首吸气声：前 0.5 s 一个先升后降的沙沙
    L = int(0.5 * sr)
    if len(y) > L:
        n2 = rng.standard_normal(L)
        n2 = sosfilt(butter(2, [300, 4000], 'bp', fs=sr, output='sos'), n2)
        n2 = n2 / (np.sqrt((n2 ** 2).mean()) + 1e-12)
        ramp = np.sin(np.linspace(0, np.pi, L)) ** 1.5
        y[:L] = y[:L] + n2 * ramp * 0.02 * np.abs(y).max()

    # 3) 句尾气散：最后 0.25 s 高频渐弱
    L2 = int(0.25 * sr)
    if len(y) > L2:
        n3 = rng.standard_normal(L2)
        n3 = sosfilt(butter(2, [2000, 6000], 'bp', fs=sr, output='sos'), n3)
        n3 = n3 / (np.sqrt((n3 ** 2).mean()) + 1e-12)
        tail = np.linspace(1, 0, L2) ** 1.2
        y[-L2:] = y[-L2:] + n3 * tail * 0.04 * np.abs(y).max()
    return y


rng = np.random.default_rng(7)
sr_out = None
pieces = []
for i, seg in enumerate(segments):
    req = {
        'text': seg, 'text_lang': 'zh',
        'ref_audio_path': ref, 'prompt_text': prompt_text, 'prompt_lang': 'zh',
        'top_k': 8, 'top_p': 1.0, 'temperature': 1.0,
        'text_split_method': 'cut5',
        'batch_size': 1, 'speed_factor': 0.8,
        'repetition_penalty': 1.35, 'sample_steps': 8,
        'parallel_infer': True, 'streaming_mode': False,
    }
    audio = None
    for sr, a in tts.run(req):
        sr_out = sr
        audio = a if audio is None else np.concatenate([audio, a])
    if audio is None:
        continue
    a = np.asarray(audio, dtype=np.float32)
    a = breath_layer(a, sr_out, rng)
    print('  seg%d %.2fs  %s' % (i, len(a) / sr_out, seg[:24]), flush=True)
    pieces.append(a)

sr = sr_out or 32000
joined = [np.zeros(int(0.3 * sr), dtype=np.float32)]
for k, p in enumerate(pieces):
    joined.append(p)
    joined.append(np.zeros(int((1.6 if k == len(pieces) - 1 else 0.85) * sr), dtype=np.float32))
y = np.concatenate(joined)

# 极短房间：只保留一点点空间感，不做大混响
rng2 = np.random.default_rng(3)
ir_len = int(0.25 * sr)
tt = np.arange(ir_len) / sr
ir = rng2.standard_normal(ir_len) * np.exp(-tt / 0.055)
ir = sosfilt(butter(2, [180, 6000], 'bp', fs=sr, output='sos'), ir)
ir = ir / np.sqrt((ir ** 2).sum())
wet = np.convolve(y, ir * 0.35, 'same')
y = y * 0.9 + wet * 0.18

y = sosfilt(butter(2, 60, 'hp', fs=sr, output='sos'), y)
y = sosfilt(butter(4, 12000, 'lp', fs=sr, output='sos'), y)
y = y / (np.abs(y).max() + 1e-9) * 0.6

out = os.path.join(here, 'out.wav')
sf.write(out, y, sr)
print('SAVED %s  %.1fs @ %d Hz' % (out, len(y) / sr, sr), flush=True)

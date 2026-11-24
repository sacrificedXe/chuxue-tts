import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

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

print('ckpt:', ckpt, flush=True)
print('ref :', ref, flush=True)

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

text = os.environ.get('TARGET_TEXT') or '假如我是一朵雪花|翩翩的在半空里潇洒|我一定认清我的方向|飞扬，飞扬，飞扬|这地面上有我的方向'
segments = [s.strip() for s in text.split('|') if s.strip()]
print('segments:', len(segments), flush=True)

sr_out = None
pieces = []
for i, seg in enumerate(segments):
    req = {
        'text': seg,
        'text_lang': 'zh',
        'ref_audio_path': ref,
        'prompt_text': prompt_text,
        'prompt_lang': 'zh',
        'top_k': 5, 'top_p': 1.0, 'temperature': 1.0,
        'text_split_method': 'cut0',
        'batch_size': 1, 'speed_factor': 0.8,
        'repetition_penalty': 1.35, 'sample_steps': 16,
        'parallel_infer': True, 'streaming_mode': False,
    }
    audio = None
    for sr, a in tts.run(req):
        sr_out = sr
        audio = a if audio is None else np.concatenate([audio, a])
    if audio is None:
        continue
    print('  seg%d %.2fs  %s' % (i, len(audio) / sr_out, seg[:20]), flush=True)
    pieces.append(np.asarray(audio, dtype=np.float32))

sr = sr_out or 32000
gap = np.zeros(int(1.0 * sr), dtype=np.float32)
lead = np.zeros(int(0.35 * sr), dtype=np.float32)
joined = [lead]
for k, p in enumerate(pieces):
    joined.append(p)
    joined.append(np.zeros(int((1.4 if k == len(pieces) - 1 else 0.9) * sr), dtype=np.float32))
y = np.concatenate(joined)

# ASMR 感：极短房间 + 轻微低频抬升 + 柔化高频
rng = np.random.default_rng(3)
ir_len = int(0.28 * sr)
tt = np.arange(ir_len) / sr
ir = rng.standard_normal(ir_len) * np.exp(-tt / 0.06)
from scipy.signal import butter, sosfilt
ir = sosfilt(butter(2, [150, 5500], 'bp', fs=sr, output='sos'), ir)
ir = ir / np.sqrt((ir ** 2).sum())
wet = np.convolve(y, ir * 0.4, 'same')
y = y * 0.88 + wet * 0.22

y = sosfilt(butter(2, 70, 'hp', fs=sr, output='sos'), y)
y = sosfilt(butter(4, 11000, 'lp', fs=sr, output='sos'), y)
y = y / (np.abs(y).max() + 1e-9) * 0.55

out = os.path.join(here, 'out.wav')
sf.write(out, y, sr)
print('SAVED %s  %.1fs @ %d Hz' % (out, len(y) / sr, sr), flush=True)

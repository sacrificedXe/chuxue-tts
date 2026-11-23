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

# 初雪模型包解压后的路径
ckpt = glob.glob(os.path.join(os.path.dirname(ROOT), 'cx', '**', '*.ckpt'), recursive=True)[0]
pth = glob.glob(os.path.join(os.path.dirname(ROOT), 'cx', '**', '*.pth'), recursive=True)[0]
ref = glob.glob(os.path.join(os.path.dirname(ROOT), 'cx', '**', '*reference_audios*', '**', '*.wav'), recursive=True)[0]
print('ckpt:', ckpt)
print('pth :', pth)
print('ref :', ref)

cfg = TTS_Config('GPT_SoVITS/configs/tts_infer.yaml')
cfg.device = 'cpu'
cfg.is_half = False
cfg.version = 'v4'
cfg.bert_base_path = 'GPT_SoVITS/pretrained_models/chinese-roberta-base'
cfg.cnhuhbert_base_path = 'GPT_SoVITS/pretrained_models/chinese-hubert-base'

tts = TTS(cfg)
tts.init_t2s_weights(ckpt)
tts.init_vits_weights(pth)

text = os.environ.get('TARGET_TEXT') or '博士，今天也没有飞行实验的排期。'
req = {
    'text': text,
    'text_lang': '中文',
    'ref_audio_path': ref,
    'prompt_text': '啊，我是现任剩女，没错。',
    'prompt_lang': '中文',
    'top_k': 5,
    'top_p': 1.0,
    'temperature': 1.0,
    'text_split_method': 'cut5',
    'batch_size': 1,
    'speed_factor': 1.0,
    'repetition_penalty': 1.35,
    'sample_steps': 16,
    'parallel_infer': True,
    'streaming_mode': False,
}

last = None
for sr, audio in tts.run(req):
    last = (sr, audio)
    print('chunk %.2fs' % (len(audio) / sr), flush=True)

sr, audio = last
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out.wav')
sf.write(out, audio, sr)
print('saved %s  %.2fs @ %d Hz' % (out, len(audio) / sr, sr))

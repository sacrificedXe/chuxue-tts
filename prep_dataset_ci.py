import glob
import os

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

SR = 32000
OUT = 'dataset'
os.makedirs(OUT + '/wavs', exist_ok=True)

x, sr0 = sf.read('cvcn.wav')
if x.ndim > 1:
    x = x.mean(1)
if sr0 != SR:
    x = librosa.resample(x.astype(np.float32), orig_sr=sr0, target_sr=SR)
print('audio %.1fs' % (len(x) / SR), flush=True)

win = int(0.03 * SR)
hop = int(0.01 * SR)
frames = [(i, float(np.sqrt((x[i:i + win] ** 2).mean()))) for i in range(0, len(x) - win, hop)]
e = np.array([f[1] for f in frames])
thr = max(0.008, np.percentile(e, 25) * 0.55)
v = e > thr

segs = []
i = 0
n = len(v)
while i < n:
    if v[i]:
        j = i
        sil = 0
        while j < n and sil < 40:          # 400 ms 静音断开
            sil = 0 if v[j] else sil + 1
            j += 1
        a = frames[i][0]
        b = min(len(x), frames[min(j, n - 1)][0] + win)
        if (b - a) / SR > 1.0:
            segs.append([a, b])
        i = j
    else:
        i += 1

merged = [segs[0]]
for a, b in segs[1:]:
    if (b - a) / SR < 2.0 and (b - merged[-1][0]) / SR < 9.0:
        merged[-1][1] = b
    else:
        merged.append([a, b])
d = np.array([(b - a) / SR for a, b in merged])
print('segments %d  total %.1fs  median %.2fs' % (len(merged), d.sum(), np.median(d)), flush=True)

for k, (a, b) in enumerate(merged):
    sf.write('%s/wavs/%03d.wav' % (OUT, k), x[a:b].astype(np.float32), SR)

model = WhisperModel('small', device='cpu', compute_type='int8')
lines = []
for f in sorted(glob.glob(OUT + '/wavs/*.wav')):
    segs_asr, _ = model.transcribe(f, language='zh', beam_size=5, vad_filter=False)
    t = ''.join(s.text for s in segs_asr).strip()
    if len(t) < 2:
        continue
    lines.append('%s|云迹|zh|%s' % (f, t))
    print(os.path.basename(f), '|', t[:48], flush=True)

open(OUT + '/yuji.list', 'w', encoding='utf-8').write('\n'.join(lines))
print('LABELED', len(lines), flush=True)

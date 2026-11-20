"""Склейка озвученных фраз: parts/<name>/NNN.wav → tts_en/out/<name>.mp3 (+ чистка tts/clean.py)
и tts_en/out/<name>.json — где в файле начинается и кончается каждая фраза (по нему ролик раскладывает голос
точно, без поиска пауз). Между фразами пауза GAP.

python tts_en/join.py <name>
"""
import glob, json, os, shutil, subprocess, sys
D = os.environ.get("TTS_DIR", "tts_en")
import numpy as np
import soundfile as sf

GAP = 0.9
name = sys.argv[1]
lines = [l.strip() for l in open(f"{D}/{name}.txt", encoding="utf-8") if l.strip()]
files = sorted(glob.glob(f"parts/{name}/*.wav"))
assert len(files) == len(lines), f"{name}: фраз {len(lines)}, файлов {len(files)}"
audio, timings, t, sr = [], [], 0.0, None
for i, f in enumerate(files):
    x, sr_ = sf.read(f, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    sr = sr or sr_
    audio += [x, np.zeros(int(sr * GAP), np.float32)]
    d = len(x) / sr
    timings.append({"n": i + 1, "start": round(t, 3), "end": round(t + d, 3), "text": lines[i]})
    t += d + GAP
os.makedirs(f"{D}/out", exist_ok=True)
sf.write(f"/tmp/{name}.wav", np.concatenate(audio[:-1]), sr)
subprocess.run([sys.executable, "tts/clean.py", f"/tmp/{name}.wav", f"{D}/out/{name}.mp3"], check=True)
json.dump(timings, open(f"{D}/out/{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
shutil.copy(f"{D}/{name}.txt", f"{D}/out/{name}.src")
print("ok", name, f"{t:.1f} c, фраз {len(lines)}")

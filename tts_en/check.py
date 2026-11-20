"""Проверка озвучки распознаванием речи (faster-whisper): каждая фраза из tts_en/out/<name>.json сверяется с текстом.
Числа при сравнении не учитываются (их видно в строке «услышано»).
Пишет tts_en/out/<name>.check.txt: доля совпавших слов, пропущенное и лишнее, что услышано.
«!!» — совпадение ниже 80% (проверить на слух: часто это просто имя, которое распознавание пишет иначе),
«~» — после текста есть лишние слова («хвост»), указано, где обрезать.

python tts_en/check.py <name>
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr

name = sys.argv[1]
D = os.environ.get("TTS_DIR", "tts_en")
tm = json.load(open(f"{D}/out/{name}.json", encoding="utf-8"))
SR = 16000
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{D}/out/{name}.mp3", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
model = asr.load(os.environ.get("TTS_LANG", "en"))
out, bad = [], []
for x in tm:
    a = audio[int(max(0, x["start"] - 0.1) * SR): int((x["end"] + 0.25) * SR)]
    r = asr.analyze(model, a, x["text"])
    flag = "!!" if r["score"] < 0.8 else ("~ " if r["junk"] else "  ")
    if flag != "  ":
        bad.append(x["n"])
    out.append(f"{flag} {x['n']:3d} {r['score']:5.0%}  {x['end'] - x['start']:5.1f} c\n    текст:   {x['text']}\n    услышано: {r['heard']}\n"
               + (f"    пропущено: {' '.join(r['miss'])}\n" if r["miss"] else "")
               + (f"    лишнее: {' '.join(r['extra'])}\n" if r["extra"] else "")
               + (f"    хвост: «{' '.join(r['junk'])}» — обрезать на {r['cut'] - 0.1:.2f} c от начала фразы\n" if r["junk"] else ""))
    print(flag, x["n"], f"{r['score']:.0%}", flush=True)
head = f"{name}: фраз {len(tm)}, проверить: {' '.join(map(str, bad)) or 'нет'}\n\n"
open(f"{D}/out/{name}.check.txt", "w", encoding="utf-8").write(head + "\n".join(out))
print(head)

"""Матрица задач для .github/workflows/tts_en.yml: каждый tts_en/<name>.txt делится на куски ~1 минуты речи
(≈ 900 знаков; минута речи на CPU озвучивается ~10–14 мин), куски идут параллельно.
Текст, который уже озвучен (tts_en/out/<name>.src совпадает с текстом), пропускается.
Печатает строки для $GITHUB_OUTPUT: matrix=…, names=…"""
import glob, json, math, os
D = os.environ.get("TTS_DIR", "tts_en")

PER_JOB = 900
include, names = [], []
for path in sorted(glob.glob(f"{D}/*.txt")):
    name = os.path.basename(path)[:-4]
    if name == "stress":                 # словарь ударений, не текст
        continue
    text = open(path, encoding="utf-8").read()
    done = f"{D}/out/{name}.src"
    if os.path.exists(done) and open(done, encoding="utf-8").read() == text and os.path.exists(f"{D}/out/{name}.mp3"):
        continue
    lines = [l for l in text.splitlines() if l.strip()]
    parts = max(1, min(len(lines), math.ceil(sum(map(len, lines)) / PER_JOB)))
    names.append(name)
    include += [{"name": name, "part": p, "parts": parts} for p in range(parts)]
print("matrix=" + json.dumps({"include": include or [{"name": "", "part": 0, "parts": 0}]}))
print("names=" + " ".join(names))
print("any=" + ("1" if names else ""))

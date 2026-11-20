"""Озвучка встроенным голосом (английский; русский — TTS_LANG=ru, с ударениями) Chatterbox Multilingual (MIT), без образца. Запускается на GitHub Actions.

python tts_en/run_en.py tts_en/<name>.txt <part> <parts> parts/<name>
<name>.txt — одна фраза на строку; годы и числа пишите словами (three twenty-five): цифры модель читает с ошибками. Фразы делятся на <parts> подряд идущих кусков примерно поровну по длине;
этот запуск озвучивает кусок <part> и пишет каждую фразу в parts/<name>/NNN.wav (NNN — номер фразы с 0).
Длинная фраза озвучивается по предложениям и склеивается с короткой паузой — так модель не обрывает и не путает текст.
Каждая озвученная фраза распознаётся (asr.py): лишний «хвост» после текста обрезается, плохая фраза
переозвучивается другой попыткой (до 3, берётся лучшая); журнал — parts/<name>/log-<part>.txt.
Параметры голоса — docs/VOICES.md: language_id="en", exaggeration=0.45, cfg_weight=0.5, temperature=0.6.
"""
import os, re, sys
import torch, torchaudio as ta

LANG = os.environ.get("TTS_LANG", "en")   # en | ru
_load = torch.load        # веса сохранены для видеокарты — грузим на CPU
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": torch.device("cpu")})
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MAX = 200                 # длиннее — режем на предложения
INNER_GAP = 0.28          # пауза между предложениями внутри фразы


def chunks(lines, parts):
    total = sum(len(l) for l in lines)
    bounds, acc, k = [0], 0, 1
    for i, l in enumerate(lines):
        acc += len(l)
        if k < parts and acc >= total * k / parts:
            bounds.append(i + 1); k += 1
    while len(bounds) < parts + 1:
        bounds.append(len(lines))
    bounds[-1] = len(lines)
    return bounds


def pieces(line):
    if len(line) <= MAX:
        return [line]
    sent = re.split(r"(?<=[.:;?!])\s+", line)
    out, cur = [], ""
    for s in sent:
        if cur and len(cur) + len(s) + 1 > MAX:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    res = []                                  # одно предложение всё ещё длинное — по запятым
    for p in out:
        if len(p) <= MAX * 1.3:
            res.append(p); continue
        cur = ""
        for s in re.split(r"(?<=,)\s+", p):
            if cur and len(cur) + len(s) + 1 > MAX:
                res.append(cur); cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            res.append(cur)
    return res


src, part, parts, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lines = [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
b = chunks(lines, parts)
idx = range(b[part], b[part + 1])
os.makedirs(out_dir, exist_ok=True)
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
gap = torch.zeros(1, int(model.sr * INNER_GAP))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr
whisper = asr.load(LANG)

ACUTE = "\u0301"
if LANG == "ru":
    # ударения: словарь (tts_ru/stress.txt, знак + перед гласной) + автомат russian_text_stresser;
    # в модель идёт размеченный текст, её собственная разметка отключается (как в tts/run.py)
    from russian_text_stresser.text_stresser import RussianTextStresser
    stresser = RussianTextStresser()
    DICT = {}
    for path in ("tts_ru/stress.txt",):
        for w in open(path, encoding="utf-8"):
            w = w.strip()
            if w and not w.startswith("#") and "+" in w:
                DICT[w.replace("+", "").lower()] = w.lower()
    try:
        from chatterbox.models.tokenizers import tokenizer as tk
        if hasattr(tk, "add_russian_stress"):
            tk.add_russian_stress = lambda t: t
    except ImportError:
        pass


def stress(line):
    if LANG != "ru":
        return line
    def rep(m):
        word = m.group(0)
        if "+" in word or word.lower() not in DICT:
            return word
        new = DICT[word.lower()]
        return new[0].upper() + new[1:] if word[0].isupper() else new
    line = re.sub(r"[\w+]+", rep, line)
    manual = {}
    def keep(m):
        key = f"QW{len(manual)}Z"
        manual[key] = re.sub(r"\+(.)", lambda v: v.group(1) + ACUTE, m.group(0))
        return key
    line = re.sub(r"\w*\+\w+", keep, line)
    out = stresser.stress_text(line)
    for k, v in manual.items():
        out = out.replace(k, v)
    return out
TRIES, GOOD = 3, 0.8
log = open(os.path.join(out_dir, f"log-{part}.txt"), "w", encoding="utf-8")


def say(text, seed):
    """Озвучить кусок; распознать; обрезать «хвост» (модель иногда договаривает бессмыслицу после текста)."""
    torch.manual_seed(seed)
    w = model.generate(stress(text), language_id=LANG, exaggeration=0.45, cfg_weight=0.5, temperature=0.6).cpu()
    a16 = ta.functional.resample(w, model.sr, 16000)[0].numpy()
    r = asr.analyze(whisper, a16, text)
    if r["cut"] is not None:
        n = int(r["cut"] * model.sr)
        w = w[:, :n].clone()
        f = int(0.03 * model.sr); w[:, -f:] *= torch.linspace(1, 0, f)
    return w, r


for i in idx:
    segs = []
    for k, p in enumerate(pieces(lines[i])):
        best = None
        for t in range(TRIES):                 # плохо вышло — другая попытка, берём лучшую
            w, r = say(p, 1000 + i * 10 + k + 100 * t)
            log.write(f"{i + 1}.{k} попытка {t + 1}: {r['score']:.0%}" + (f", обрезан хвост «{' '.join(r['junk'])}»" if r["junk"] else "")
                      + f"\n   услышано: {r['heard']}\n")
            if best is None or r["score"] > best[1]["score"]:
                best = (w, r)
            if r["score"] >= GOOD and len(r["junk"]) <= 12:
                break
        segs += [best[0], gap]
    wav = torch.cat(segs[:-1], dim=1)
    ta.save(os.path.join(out_dir, f"{i:03d}.wav"), wav, model.sr)
    log.flush()
    print(f"[{i + 1}/{len(lines)}] {wav.shape[-1] / model.sr:.1f} c  {lines[i][:70]}", flush=True)
log.close()
print("ok", out_dir, list(idx))

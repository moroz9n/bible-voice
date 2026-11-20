"""Сверка озвученной фразы с текстом распознаванием речи (faster-whisper). Общая часть run_en.py и check.py.

analyze(model, audio16k, text) → dict(score, heard, miss, extra, cut, junk):
  score — доля слов текста, которые услышаны (числа не считаются: модель и распознавание пишут их по-разному);
  cut   — секунда, где обрезать фразу, если модель после текста «договорила» лишнее (иначе None).
"""
import difflib, re

NUMW = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand and oh "
           "first second third fourth fifth sixth seventh i ii iii iv v vi vii".split())


# русские числительные (в тексте — словами, распознавание пишет цифрами): по началу слова
RU_NUM = tuple(("ноль один одн два две двух двум трет три четыр пят шест сем восем девят десят одиннадцат двенадцат "
          "двадцат тридцат сорок пятьдесят шестьдесят семьдесят восемьдесят девяност двест "
          "трист четырест пятисот шестисот семисот восьмисот тысяч перв втор год").split())
RU_EXACT = {"сто", "ста", "сотый", "сотом", "й", "м", "го", "х", "ом", "ый"}
LANG = "en"


def words(s):
    s = s.lower().replace("’", "'").replace("–", " ").replace("—", " ").replace("-", " ").replace("\u0301", "").replace("+", "")
    if LANG == "ru":
        s = s.replace("ё", "е")
        w = [x for x in re.findall(r"[а-яa-z0-9]+", s) if not re.search(r"\d", x)]
        return [x for x in w if not x.startswith(RU_NUM) and x not in RU_EXACT]
    w = [x for x in re.findall(r"[a-z0-9']+", s) if not x.isdigit() and x not in NUMW and not re.fullmatch(r"\d+(st|nd|rd|th)", x)]
    return ["saint" if x == "st" else x for x in w]


def load(lang="en"):
    global LANG
    LANG = lang
    from faster_whisper import WhisperModel
    return WhisperModel("small.en" if lang == "en" else "small", device="cpu", compute_type="int8")


def analyze(model, audio, text):
    segs, _ = model.transcribe(audio, language=LANG, beam_size=5, vad_filter=False, word_timestamps=True,
                               condition_on_previous_text=False)
    segs = list(segs)
    heard = " ".join(s.text.strip() for s in segs)
    ws = [w for s in segs for w in (s.words or [])]
    # слова распознавания по порядку, с временем; числа выкидываем так же, как в тексте
    hw = [(x, w.start, w.end) for w in ws for x in words(w.word)]
    t = words(text)
    h = [x for x, _, _ in hw]
    sm = difflib.SequenceMatcher(None, t, h, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    score = sum(b.size for b in blocks) / max(1, len(t))
    miss, extra = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("delete", "replace"):
            miss += t[i1:i2]
        if op in ("insert", "replace"):
            extra += h[j1:j2]
    cut, junk = None, []
    if blocks:
        last = blocks[-1]
        end_t = last.a + last.size            # сколько слов текста покрыто до конца последнего совпадения
        jh = last.b + last.size                # первое слово распознавания после совпадения
        # хвост: после последнего совпавшего слова идут слова, а текст при этом уже кончился (или почти)
        # последнее слово текста не узнано (имя: Eutyches → «Uticus», Исавре → «и савре»): ищем среди следующих
        # двух слов похожее на него — оно и есть последнее слово; хвост — только то, что после него
        if jh < len(hw) and end_t == len(t) - 1:
            best = (0.0, None)
            for k in range(jh, min(jh + 2, len(hw))):
                for n in (1, 2, 3):                      # слово могло распасться на части: «до ныне», «и савре»
                    if k + n <= len(hw):
                        r = difflib.SequenceMatcher(None, "".join(h[k:k + n]), t[-1]).ratio()
                        if r > best[0] + 0.05:
                            best = (r, k + n)
            if best[0] >= 0.5:
                jh, end_t = best[1], len(t)
        if jh < len(hw) and end_t == len(t):
            junk = h[jh:]
            last_end = hw[jh - 1][2]
            nxt = [w for w in ws if w.start >= last_end - 0.01 and not re.fullmatch(r"[\d.,:;\-–]+", w.word.strip())]
            start = nxt[0].start if nxt else hw[jh][1]
            cut = max(last_end + 0.05, start - 0.08)
    return dict(score=score, heard=heard, miss=miss, extra=extra, cut=cut, junk=junk)

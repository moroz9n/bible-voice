"""Чистка синтезированной озвучки: дыхание, шорохи, шум между словами.

python3 tts/clean.py in.mp3 out.mp3

1) ffmpeg: срез гула и верхов, шумоподавление.
2) дыхание и шорохи: участки от 90 мс, тише −28 дБ, без тона голоса и без «свиста» (энергия ниже
   3.5 кГц — так отличаем от с/ш) — приглушаются на 30 дБ с плавными краями, без щелчков.
3) выравнивание громкости.
"""
import subprocess, sys, numpy as np

SR = 44100
src, dst = sys.argv[1:3]


def ff(args, data=None):
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args], input=data, capture_output=True, check=True).stdout


raw = ff(["-i", src, "-af", "highpass=f=80,lowpass=f=11500,afftdn=nr=12:nf=-42:tn=1",
          "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"])
x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768

win = int(0.03 * SR)
n = len(x) // win
fr = x[: n * win].reshape(n, win)
rms = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
spec = np.abs(np.fft.rfft(fr * np.hanning(win), axis=1)) + 1e-9
freqs = np.fft.rfftfreq(win, 1 / SR)
cent = (spec * freqs).sum(1) / spec.sum(1)                        # «центр тяжести» спектра


def voicing(w):
    """Сила тона голоса 70–300 Гц: у гласных и звонких ~0.4+, у дыхания и шорохов ~0.25."""
    w = w - w.mean()
    ac = np.correlate(w, w, "full")[len(w) - 1:]
    return ac[SR // 300: SR // 70].max() / ac[0] if ac[0] > 0 else 0.0


voc = np.array([voicing(w) for w in fr])

# дыхание/шорох: тихо, без тона, энергия не в свистящих (с, ш — выше 3.5 кГц)
cand = (rms < -28) & (voc < 0.33) & (cent < 3500)
cand |= rms < -52                                                  # фоновый шум — всегда
# оставляем только участки от 90 мс: короткие глухие согласные (п, т, к) не трогаем
b = np.zeros(n, bool)
i = 0
while i < n:
    if cand[i]:
        j = i
        while j < n and cand[j]:
            j += 1
        if j - i >= 3:
            b[i:j] = True
        i = j
    else:
        i += 1

gain_w = np.where(b, 10 ** (-30 / 20), 1.0)
gain = np.repeat(gain_w, win)
gain = np.convolve(gain, np.hanning(int(0.02 * SR)) / np.hanning(int(0.02 * SR)).sum(), "same")
y = x[: len(gain)] * gain
y = np.concatenate([y, x[len(gain):]])

print(f"приглушено как дыхание/шорох: {b.mean() * 100:.1f}% времени ({b.sum() * 0.03:.1f} c)")
pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
ff(["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", "deesser=i=0.3,loudnorm=I=-16:TP=-1.5",
    "-ar", str(SR), "-b:a", "160k", dst], pcm)
print("ok", dst)

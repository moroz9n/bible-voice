# bible-voice

Озвучка роликов каналов по Библии ИИ-голосом: Chatterbox Multilingual (MIT), встроенный голос без образца,
`exaggeration=0.45, cfg_weight=0.5, temperature=0.6`, затем чистка `tts/clean.py`. Проверка распознаванием (faster-whisper).

- `tts_ru/<name>.txt` — русские фразы (одна на строку, числа словами), ударения — `tts_ru/stress.txt`.
- `tts_en/<name>.txt` — английские фразы.
- Пуш текста запускает GitHub Actions; результат — `tts_*/out/<name>.mp3` и `<name>.json` (время каждой фразы).

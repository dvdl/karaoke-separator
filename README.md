# karaoke-separator

Runpod serverless worker that separates a song into instrumental + vocals for
karaoke.dvdlev.com, with **becruily deux** (Mel-Band RoFormer).

- Model: [becruily/mel-band-roformer-deux](https://huggingface.co/becruily/mel-band-roformer-deux)
  by becruily, **CC-BY-NC-4.0**. The weights are *not* in this repository or image; the worker
  downloads them from Hugging Face at cold start. Non-commercial use only.
- Runtime: [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.47 (MIT).
- Image: `ghcr.io/dvdl/karaoke-separator` (built by GitHub Actions, code + libraries only).

Input `{"audio_url"|"audio_b64", "upload": {"instrumental": PUT url, "vocals": PUT url}}`;
see `handler.py`. `ab/` holds the scripts of the 2026-10-14 A/B run (short-lived pod).

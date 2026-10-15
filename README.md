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

## sw6 (6-stem separation)

- Model: BS-RoFormer SW 6-stem ([FractalSurfer/BS-RoFormer-SW-6stem](https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem)).
- Licence: has **NO stated licence** for its weights -> private/personal use only, not redistributed, fetched at cold start.
- GPU pool: the endpoint is pinned to the `AMPERE_24` pool (RTX 3090/A5000/L4) because "-<MIG type>" pool exclusions were not honoured on 2026-10-14, and the handler refuses MIG as a second guard.


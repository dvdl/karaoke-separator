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

## vocsplit (lead / backing vocal separation)

- Model: becruily Mel-Band RoFormer Karaoke ([becruily/mel-band-roformer-karaoke](https://huggingface.co/becruily/mel-band-roformer-karaoke)).
- Licence: has **NO stated licence** for its weights -> private/personal use only, not baked into the image, fetched at first use on demand.
- Task: `{"task": "vocsplit", "audio_b64": "<base64 FLAC>"}` (also accepts `audio_url`).
- Output: `{"task": "vocsplit", "model": "becruily-karaoke", "gpu": str, "audio_seconds": float, "separate_seconds": float, "load_seconds": float, "lead_b64": str, "backing_b64": str}` (both stems FLAC PCM_24).


## Runpod API v2 migration (2026-10-20)

REST v1 (rest.runpod.io/v1) is retired 2026-11-15; GraphQL is rate-limited from 2026-10-27 and stops in January 2027.
`endpoint.py`, `ab/pod.py` and `ab/pod_setup.sh` use REST v2 (`https://api.runpod.io/v2`).

The untracked one-off scripts in the main checkout are **marked for retirement**; do not use them after 2026-11-15:

| Script | Uses | Replacement |
|---|---|---|
| `ep6.py` `bal` | GraphQL `myself.clientBalance` | `runpodctl user` or the console (no v2 balance endpoint) |
| `ep6.py` `tmpl` / `image` / `scale` | v1 field names (`imageName`, `/endpoints`, `workersMax`) via `endpoint.py` | `GET`/`PATCH /v2/templates/{id}` with `image`; `endpoint.py on/off` |
| `ep6gpus.py` | GraphQL `saveEndpoint gpuIds` | `endpoint.py gpus` (v2 `gpu.pools` + `excludedTypes`) |
| `ep6probe.py` | GraphQL `myself`/`endpoints`/`gpuTypes` | `endpoint.py status`, `GET /v2/catalog/gpus` |
| `ab/work/six/ep6.py`, `ab/work/rearm.sh` (git-ignored scratch) | copy of ep6 / runpodctl self-kill | `ab/pod_setup.sh` |

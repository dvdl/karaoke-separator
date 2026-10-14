"""Runpod serverless worker: becruily deux (Mel-Band RoFormer) vocal/instrumental separation.

The weights are NOT in the image (CC-BY-NC-4.0, not ours to republish): they are fetched
from Hugging Face once per cold start, before the worker takes jobs, so FlashBoot snapshots
a loaded model.

Input  {"audio_url": "https://..."}  or  {"audio_b64": "<base64 audio>"}
       optional "upload": {"instrumental": PUT url, "vocals": PUT url}
         (song-length stems exceed Runpod's payload limit; the site hands out signed URLs)
Output {"model": "becruily-deux", "gpu": str, "audio_seconds": float, "separate_seconds": float,
        "instrumental_b64"/"vocals_b64" (FLAC) unless uploaded, "uploaded": bool}
"""
import base64, os, shutil, tempfile, time, urllib.request

import soundfile as sf
import torch
import runpod
from audio_separator.separator import Separator

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
HF = "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main"
CKPT, YAML = "becruily_deux.ckpt", "config_deux_becruily_mel_band_roformer.yaml"
UA = {"User-Agent": "karaoke-separator/1.0"}
OUT = "/tmp/sep_out"


def fetch(url, dst, timeout=600):
    tmp = dst + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dst)


def load():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t = time.time()
    for name, src in ((CKPT, f"{HF}/becruily_deux.ckpt"), (YAML, f"{HF}/config_deux_becruily.yaml")):
        p = os.path.join(MODEL_DIR, name)
        if not os.path.exists(p):
            fetch(src, p)
    t_dl = time.time() - t
    s = Separator(model_file_dir=MODEL_DIR, output_dir=OUT, output_format="FLAC")
    # audio-separator 0.47 has no registry entry for deux: hand it the local files.
    s.download_model_files = lambda f: (CKPT, "MDXC", "becruily deux", os.path.join(MODEL_DIR, CKPT), YAML)
    s.load_model(CKPT)
    print(f"model ready: download {t_dl:.1f}s, total {time.time()-t:.1f}s, "
          f"cuda={torch.cuda.is_available()}", flush=True)
    return s


SEP = load()
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def put(url, path):
    with open(path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={**UA, "Content-Type": "audio/flac"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if r.status >= 300:
            raise RuntimeError(f"upload {r.status}")


def handler(job):
    inp = job.get("input") or {}
    work = tempfile.mkdtemp(prefix="job_")
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    try:
        src = os.path.join(work, "input")
        if inp.get("audio_url"):
            fetch(inp["audio_url"], src)
        elif inp.get("audio_b64"):
            with open(src, "wb") as f:
                f.write(base64.b64decode(inp["audio_b64"]))
        else:
            return {"error": "need audio_url or audio_b64"}
        try:
            info = sf.info(src)
        except Exception:
            info = None
        t = time.time()
        files = SEP.separate(src)
        dt = time.time() - t
        stems = {}
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(OUT, f)
            stems["vocals" if "vocal" in os.path.basename(p).lower() else "instrumental"] = p
        if set(stems) != {"vocals", "instrumental"}:
            return {"error": f"unexpected stems {list(stems)}"}
        out = {"model": "becruily-deux", "gpu": GPU, "separate_seconds": round(dt, 2),
               "audio_seconds": round(info.duration, 2) if info else None}
        up = inp.get("upload") or {}
        if up.get("instrumental") and up.get("vocals"):
            for k in ("instrumental", "vocals"):
                put(up[k], stems[k])
            out["uploaded"] = True
        else:
            for k in ("instrumental", "vocals"):
                with open(stems[k], "rb") as f:
                    out[k + "_b64"] = base64.b64encode(f.read()).decode()
            out["uploaded"] = False
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

#!/usr/bin/env python3
"""Separate audio files with deux and/or viperx via audio-separator 0.47.
Usage: separate_ab.py --models deux,viperx --model-dir DIR --out DIR in1.wav [in2.wav ...]
Writes OUT/<stem>/<model>_{instrumental,vocals}.flac and OUT/<stem>/<model>.json (seconds).
Skips outputs that already exist (resumable on a vanished pod)."""
import argparse, json, os, sys, time, glob, shutil
from audio_separator.separator import Separator

DEUX = ("becruily_deux.ckpt", "config_deux_becruily_mel_band_roformer.yaml")
VIPERX = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
HQ4 = "UVR-MDX-NET-Inst_HQ_4.onnx"  # arm A: what the site uses now

def make(model, model_dir, tmp):
    s = Separator(model_file_dir=model_dir, output_dir=tmp, output_format="FLAC")
    if model == "deux":
        ck, ym = DEUX
        # 0.47 has no registry entry for deux: hand it the local files directly.
        s.download_model_files = lambda f: (ck, "MDXC", "becruily deux", os.path.join(model_dir, ck), ym)
        s.load_model(ck)
    elif model == "hq4":
        s.load_model(HQ4)
    else:
        s.load_model(VIPERX)
    return s

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", default="hq4,deux,viperx"); a.add_argument("--model-dir", required=True)
    a.add_argument("--out", required=True); a.add_argument("inputs", nargs="+")
    o = a.parse_args()
    for model in o.models.split(","):
        tmp = os.path.join(o.out, "_tmp_" + model); os.makedirs(tmp, exist_ok=True)
        sep = None
        for inp in o.inputs:
            stem = os.path.splitext(os.path.basename(inp))[0]
            dst = os.path.join(o.out, stem); os.makedirs(dst, exist_ok=True)
            if os.path.exists(os.path.join(dst, model + ".json")):
                continue
            sep = sep or make(model, o.model_dir, tmp)
            t = time.time(); files = sep.separate(inp); dt = time.time() - t
            for f in files:
                p = f if os.path.isabs(f) else os.path.join(tmp, f)
                kind = "vocals" if "vocal" in os.path.basename(p).lower() else "instrumental"
                shutil.move(p, os.path.join(dst, f"{model}_{kind}.flac"))
            json.dump({"model": model, "seconds": round(dt, 1)}, open(os.path.join(dst, model + ".json"), "w"))
            print(f"{model} {stem} {dt:.1f}s", flush=True)

if __name__ == "__main__":
    main()

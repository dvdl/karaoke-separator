#!/usr/bin/env python3
"""After the run (on the box): metrics + blind clips.
Inputs: IN/<id>.wav (original mix as separated), OUT/<id>/<arm>_{instrumental,vocals}.flac,
        LYR/<id>.json (the site's lyrics.json, for sung/unsung regions; optional).
Writes  RESULTS/results.csv, RESULTS/clips/<id>/{A,B,C}.mp3, RESULTS/KEY.json (keep from David).
Metrics per song x arm:
  lufs_inst          integrated loudness of the instrumental (ffmpeg ebur128)
  intro_db           instrumental minus original RMS over the first 30 s (0 = nothing lost)
  voc_unsung_db      vocal-stem RMS where no lyric line is near, minus where lines are
                     (high = instruments in the vocal stem; the צל כבד guitar shows here)
  heard_words        words faster-whisper ("small") hears in the instrumental (vocal residue)
  sep_seconds        GPU separation time
"""
import argparse, csv, json, os, random, re, subprocess
import numpy as np, soundfile as sf

ARMS = ["hq4", "deux", "viperx"]
INTRO_SONGS = {"Oj_PB40UC84"}  # צל כבד: separate intro clip for the guitar-leak check

def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)

def lufs(p):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", p, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r); return float(m[-1]) if m else None

def sung_mask(lyr, n, sr):
    m = np.zeros(n, bool)
    for ln in (lyr or {}).get("lines", []):
        s, e = ln.get("start"), ln.get("end")
        if s is None or e is None: continue
        m[max(0, int((s - 1) * sr)):min(n, int((e + 1) * sr))] = True
    return m

def main():
    a = argparse.ArgumentParser()
    for k in ("inp", "out", "lyr", "results"): a.add_argument("--" + k, required=k != "lyr")
    a.add_argument("--clip-seconds", type=float, default=60); a.add_argument("--no-whisper", action="store_true")
    o = a.parse_args(); os.makedirs(o.results, exist_ok=True)
    wm = None
    if not o.no_whisper:
        from faster_whisper import WhisperModel
        wm = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
    rows, key, rng, clip_info = [], {}, random.Random(20261014), {}
    for sid in sorted(os.listdir(o.out)):
        d = os.path.join(o.out, sid)
        if sid.startswith("_") or not os.path.isdir(d): continue
        orig, sr = sf.read(os.path.join(o.inp, sid + ".wav"))
        lyr = None
        if o.lyr and os.path.exists(os.path.join(o.lyr, sid + ".json")):
            lyr = json.load(open(os.path.join(o.lyr, sid + ".json")))
        # clip: 60 s from ~2 s before a sung line, chosen so the window is fullest of singing
        dur = len(orig) / sr; start = dur * 0.3
        lines = [l for l in (lyr or {}).get("lines", []) if l.get("start") is not None and l.get("end") is not None]
        if lines:
            def sung(t0):
                return sum(max(0, min(l["end"], t0 + o.clip_seconds) - max(l["start"], t0)) for l in lines)
            cands = [max(0, l["start"] - 2) for l in lines if l["start"] - 2 + o.clip_seconds <= dur]
            if cands: start = max(cands, key=sung)
        intro = None
        if sid in INTRO_SONGS and lines:
            intro = max(5.0, min(l["start"] for l in lines) - 0.5)   # צל כבד: before the first sung line
        clip_info[sid] = {"start": round(start, 2), "seconds": o.clip_seconds, "intro_seconds": intro}
        arms = [x for x in ARMS if os.path.exists(os.path.join(d, x + "_instrumental.flac"))]
        labels = rng.sample("ABC"[:len(arms)], len(arms)); key[sid] = {}
        cd = os.path.join(o.results, "clips", sid); os.makedirs(cd, exist_ok=True)
        for arm, lab in zip(arms, labels):
            ip, vp = (os.path.join(d, f"{arm}_{k}.flac") for k in ("instrumental", "vocals"))
            inst, _ = sf.read(ip); voc, _ = sf.read(vp) if os.path.exists(vp) else (None, None)
            n = min(len(inst), len(orig))
            first = min([l['start'] for l in (lyr or {}).get('lines', []) if l.get('start')] or [30])
            i30 = min(n, int(max(5, first - 0.5) * sr))  # intro = before the first sung line
            row = {"song": sid, "arm": arm, "lufs_inst": lufs(ip),
                   "intro_db": round(rms_db(inst[:i30]) - rms_db(orig[:i30]), 2)}
            if voc is not None and lyr:
                m = sung_mask(lyr, len(voc), sr)
                if m.any() and (~m).any():
                    row["voc_unsung_db"] = round(rms_db(voc[~m]) - rms_db(voc[m]), 2)
            if wm:
                pcm = np.frombuffer(subprocess.run(["ffmpeg","-loglevel","error","-i",ip,"-ac","1","-ar","16000","-f","s16le","-"],capture_output=True,check=True).stdout, np.int16).astype(np.float32) / 32768
                segs, _ = wm.transcribe(pcm, vad_filter=True, condition_on_previous_text=False)
                row["heard_words"] = sum(len(s.text.split()) for s in segs)
            js = os.path.join(d, arm + ".json")
            if os.path.exists(js): row["sep_seconds"] = json.load(open(js)).get("seconds")
            rows.append(row); key[sid][lab] = arm
            def cut(t0, secs, name):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", ip, "-af",
                                f"atrim=start={t0:.2f}:duration={secs},asetpts=PTS-STARTPTS,loudnorm=I=-16:TP=-1.5",
                                "-ar", "44100", "-b:a", "192k", os.path.join(cd, name)], check=True)
            cut(start, o.clip_seconds, f"{lab}.mp3")
            if intro: cut(0, intro, f"intro_{lab}.mp3")
            print(row, flush=True)
    with open(os.path.join(o.results, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("song","arm"), k)))
        w.writeheader(); w.writerows(rows)
    json.dump(key, open(os.path.join(o.results, "KEY.json"), "w"), indent=1)
    json.dump(clip_info, open(os.path.join(o.results, "clip_times.json"), "w"), indent=1)

if __name__ == "__main__":
    main()

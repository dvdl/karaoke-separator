#!/usr/bin/env python3
"""Runpod serverless endpoint for deux: create | status | gpus | test FILE | off | on.
Key from ~/ClaudeCode/maaravi-analytics/.env (RUNPOD_API_KEY, never printed).
Guards: workersMax 1, workersMin 0, idleTimeout 5 s, executionTimeout 300 s;
`off` sets workersMax 0 (the kill switch)."""
import base64, json, re, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = Path.home() / "ClaudeCode/maaravi-analytics/.env"


def get_key():
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", ENV_FILE.read_text(), re.M))["RUNPOD_API_KEY"].strip().strip('"')


def __getattr__(name):
    if name == "KEY":
        return get_key()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


IDFILE = ROOT / ".endpoint_id"
IMAGE = "ghcr.io/dvdl/karaoke-separator:b5c0e162452d123bd5af4a48c7a77cb1d0cd5f90"
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000",
        "NVIDIA RTX A4500", "NVIDIA L4"]   # fast, full (non-MIG) GPUs only
# REST gpuTypeIds maps to whole pools (AMPERE_16/AMPERE_24/ADA_24), which also hold slow
# cards and the "RTX PRO 6000 Blackwell ... MIG 1g.24gb" slice. `gpus` saves pools plus
# explicit "-<type>" exclusions so only GPUS can ever be scheduled.
POOLS = ["ADA_24", "AMPERE_24", "AMPERE_16"]
EXCLUDE = ["NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A4000",
           "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"]
UA = "karaoke-separator-admin/1.0"   # Python's default UA gets Cloudflare 1010
REST_BASE = "https://api.runpod.io/v2"


def format_error(data):
    """Format RFC 9457 Problem Details or other error responses."""
    if isinstance(data, dict):
        title = data.get("title")
        detail = data.get("detail")
        status = data.get("status")
        errors = data.get("errors")
        if title or detail or status:
            parts = []
            if status:
                parts.append(f"[{status}]")
            if title:
                parts.append(title)
            if detail:
                parts.append(f"- {detail}" if parts else detail)
            if errors and isinstance(errors, list):
                parts.append(f"({'; '.join(str(e) for e in errors)})")
            return " ".join(parts)
    return str(data)


def call(method, url, body=None, timeout=60):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {get_key()}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = raw
            return r.status, data
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw[:400]


rest = lambda m, p, b=None: call(m, REST_BASE + p, b)
eid = lambda: IDFILE.read_text().split()[0]


def unwrap(data, key=None):
    """Unwrap v2 list responses like {'endpoints': [...]}, {'workers': [...]}, {'templates': [...]}."""
    if isinstance(data, dict):
        if key and key in data and isinstance(data[key], list):
            return data[key]
        for k in ("endpoints", "workers", "templates", "pods"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return data


def create():
    if IDFILE.exists():
        sys.exit(f"exists: {eid()}")
    c, t = rest("POST", "/templates", {
        "name": "karaoke-separator-deux",
        "image": IMAGE,
        "serverless": True,
        "startJupyter": False,
        "disk": 15,
        "env": {},
    })
    print("template", c, t.get("id") if (isinstance(t, dict) and "id" in t and c in (200, 201)) else format_error(t))
    if c not in (200, 201): sys.exit(1)
    c, e = rest("POST", "/serverless", {
        "name": "karaoke-separator-deux",
        "type": "QUEUE",
        "templateId": t["id"],
        "gpu": {"pools": POOLS, "excludedTypes": EXCLUDE, "count": 1, "minCudaVersion": "12.8"},
        "workers": {"min": 0, "max": 1, "idleTimeout": 5},
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
        "timeout": 300000,
        "flashboot": "FLASHBOOT",
    })
    print("endpoint", c, e.get("id") if (isinstance(e, dict) and "id" in e and c in (200, 201)) else format_error(e))
    if c not in (200, 201): sys.exit(1)
    IDFILE.write_text(f"{e['id']} {t['id']}\n")
    gpus()


def status():
    c, e = rest("GET", f"/serverless/{eid()}")
    if c == 200 and isinstance(e, dict):
        w_cfg = e.get("workers") if isinstance(e.get("workers"), dict) else {}
        w_min = w_cfg.get("min") if "min" in w_cfg else e.get("workersMin")
        w_max = w_cfg.get("max") if "max" in w_cfg else e.get("workersMax")
        idle = w_cfg.get("idleTimeout") if "idleTimeout" in w_cfg else e.get("idleTimeout")
        exec_t = e.get("timeout") if "timeout" in e else (e.get("executionTimeoutMs") or e.get("executionTimeout"))
        print({
            "id": e.get("id"),
            "workersMin": w_min,
            "workersMax": w_max,
            "idleTimeout": idle,
            "executionTimeoutMs": exec_t,
        })
    else:
        print(format_error(e))
    cw, w = rest("GET", f"/serverless/{eid()}/workers")
    workers = unwrap(w, "workers")
    print("workers:", [(x.get("id"), x.get("status") or x.get("desiredStatus")) for x in (workers or [])] if cw == 200 else format_error(w))
    c, h = call("GET", f"https://api.runpod.ai/v2/{eid()}/health"); print("health", h if c == 200 else format_error(h))


def setmax(n):
    c, e = rest("PATCH", f"/serverless/{eid()}", {"workers": {"max": n}})
    print("workersMax ->", n, c if c in (200, 204) else f"{c} {format_error(e)}")


def gpus():
    c, r = rest("PATCH", f"/serverless/{eid()}", {
        "gpu": {"pools": POOLS, "excludedTypes": EXCLUDE, "count": 1},
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
        "workers": {"min": 0, "max": 1, "idleTimeout": 5},
        "flashboot": "FLASHBOOT",
    })
    print(c, r if c in (200, 204) else format_error(r))


def test(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    t0 = time.time()
    c, j = call("POST", f"https://api.runpod.ai/v2/{eid()}/run", {"input": {"audio_b64": b64}, "policy": {"executionTimeout": 300000, "ttl": 900000}})
    print("submit", c, j.get("id") if isinstance(j, dict) else j)
    jid = j["id"]
    while time.time() - t0 < 600:
        time.sleep(5)
        c, s = call("GET", f"https://api.runpod.ai/v2/{eid()}/status/{jid}")
        st = s.get("status") if isinstance(s, dict) else s
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            out = s.get("output") or {}
            if isinstance(out, dict):
                for k in ("instrumental_b64", "vocals_b64"):
                    if k in out:
                        (ROOT / "ab/work" / f"endpoint_test_{k[:-4]}.flac").write_bytes(base64.b64decode(out.pop(k)))
            print(st, "delayTime", s.get("delayTime"), "executionTime", s.get("executionTime"), "wall", round(time.time() - t0), out)
            return
    call("POST", f"https://api.runpod.ai/v2/{eid()}/cancel/{jid}"); print("gave up, cancelled")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"create": create, "status": status, "off": lambda: setmax(0), "on": lambda: setmax(1), "gpus": gpus,
     "test": lambda: test(sys.argv[2])}[cmd]()

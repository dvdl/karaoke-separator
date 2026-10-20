#!/usr/bin/env python3
"""RunPod pod lifecycle for the karaoke separation A/B (adapted from film-kit/launch_pod.py).
Guards: `down` is idempotent and verifies via API; `up` writes .pod_id; `watchdog` kills past
--max-minutes; the pod also arms its own timer (pod_setup.sh). Runs on David's Mac.
  up | status | down | watchdog --max-minutes N | sshcfg
"""
import argparse, json, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = Path.home() / "ClaudeCode/maaravi-analytics/.env"  # RUNPOD_API_KEY, never printed
PUBKEY_FILE = Path.home() / ".ssh/runpod_maaravi.pub"
REST = "https://api.runpod.io/v2"
PODFILE = ROOT / ".pod_id"


def get_key():
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", ENV_FILE.read_text(), re.M))["RUNPOD_API_KEY"].strip().strip('"')


def get_pubkey():
    return PUBKEY_FILE.read_text().strip()


def __getattr__(name):
    if name == "KEY":
        return get_key()
    if name == "PUBKEY":
        return get_pubkey()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 16-24 GB is plenty (three models, loaded one at a time, ~0.4-0.6 GB weights each).
# Rates are the list prices seen 2026-10 [re-check before launch]; `up` tries them in order.
GPU_CANDIDATES = [  # SECURE: has a public IP + port 22 (community 4090 never got one, 2026-10-14)
    ("NVIDIA RTX A5000", "SECURE", 0.27),
    ("NVIDIA RTX A4500", "SECURE", 0.25),
    ("NVIDIA RTX 4000 Ada Generation", "SECURE", 0.28),
    ("NVIDIA RTX A6000", "SECURE", 0.59),
    ("NVIDIA GeForce RTX 4090", "SECURE", 0.89),
]
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"  # the 2.9.1 tag does not exist
NAME = "karaoke-ab"
UA = "karaoke-ab-launcher/1.0"   # Python's default UA gets Cloudflare 1010


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


def api(method, path, body=None):
    req = urllib.request.Request(f"{REST}{path}", method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {get_key()}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
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
            return e.code, raw[:300]
    except Exception as e:  # network blip: let callers retry
        return 0, str(e)


def unwrap(data, key="pods"):
    """Unwrap v2 list responses like {'pods': [...]}, {'endpoints': [...]}."""
    if isinstance(data, dict):
        if key in data and isinstance(data[key], list):
            return data[key]
        for k in ("pods", "endpoints", "workers", "templates"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return data


def action(pid, act):
    return api("POST", f"/pods/{pid}/action", {"action": act})


def stop_pod(pid):
    return action(pid, "stop")


def start_pod(pid):
    return action(pid, "start")


def live_ids():
    code, pods = api("GET", "/pods")
    if code != 200:
        return None
    pod_list = unwrap(pods, "pods")
    return [p["id"] for p in pod_list] if isinstance(pod_list, list) else []


def cmd_up(a):
    if PODFILE.exists():
        sys.exit(f"refusing: {PODFILE} exists. Run `down` first.")
    for gpu, cloud, rate in GPU_CANDIDATES:
        body = {
            "name": NAME,
            "image": IMAGE,
            "gpu": {"id": gpu, "count": 1},
            "cloud": cloud,
            "disk": a.disk,
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": get_pubkey(), "MAX_MINUTES": str(int(a.max_minutes))},
        }
        code, res = api("POST", "/pods", body)
        if code in (200, 201):
            PODFILE.write_text(f"{res['id']}\n{gpu}\n{cloud}\n{rate}\n{time.time()}\n")
            print(f"created {res['id']} on {gpu} ({cloud}) @ ${rate}/hr"); return
        err_msg = format_error(res)
        print(f"  {gpu:<28} {cloud:<9} -> {code} {err_msg[:90]}")
    sys.exit("no capacity on any candidate GPU")


def read_pod():
    return PODFILE.read_text().split("\n") if PODFILE.exists() else None


def cmd_status(a):
    info = read_pod()
    if not info:
        print("no .pod_id; live pods:", live_ids()); return
    pid, gpu, cloud, rate, t0 = info[0], info[1], info[2], float(info[3]), float(info[4])
    code, pod = api("GET", f"/pods/{pid}")
    m = (time.time() - t0) / 60
    st = (pod.get('status') or pod.get('desiredStatus')) if (code == 200 and isinstance(pod, dict)) else format_error(pod)
    print(f"{pid} {gpu} ({cloud}) status={st} "
          f"uptime={m:.1f}min ~${m/60*rate:.2f}")


def cmd_down(a):
    info = read_pod(); pid = info[0] if info else a.pod_id
    if not pid:
        print("nothing to terminate; live pods:", live_ids()); return
    for attempt in range(5):
        code, res = api("DELETE", f"/pods/{pid}")
        err_str = f" ({format_error(res)})" if code not in (200, 204) else ""
        print(f"DELETE {pid} -> {code}{err_str}")
        time.sleep(5)
        live = live_ids()
        if live is not None and pid not in live:
            print(f"confirmed terminated. live pods now: {live}")
            if info:
                m = (time.time() - float(info[4])) / 60
                print(f"session {m:.1f} min ~${m/60*float(info[3]):.2f}")
                PODFILE.unlink()
            return
        time.sleep(15)
    sys.exit(f"!! {pid} STILL LIVE - terminate manually at runpod.io")


def cmd_sshcfg(a):
    pid = (read_pod() or [a.pod_id])[0]
    for _ in range(60):
        code, pod = api("GET", f"/pods/{pid}")
        ssh_direct = (pod.get("ssh") or {}).get("direct") if (code == 200 and isinstance(pod, dict)) else None
        ip = ssh_direct.get("host") if ssh_direct else ((pod or {}).get("publicIp") if code == 200 else None)
        port = ssh_direct.get("port") if ssh_direct else (((pod or {}).get("portMappings") or {}).get("22") if code == 200 else None)
        if ip and port:
            cfg = ROOT / "work/ssh_cfg"; cfg.parent.mkdir(exist_ok=True)
            cfg.write_text(f"Host pod\n  HostName {ip}\n  Port {port}\n  User root\n"
                           "  IdentityFile ~/.ssh/runpod_maaravi\n  StrictHostKeyChecking no\n"
                           "  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n  ServerAliveInterval 30\n")
            for _ in range(12):
                if subprocess.run(["ssh", "-F", str(cfg), "pod", "true"], capture_output=True).returncode == 0:
                    print(f"ssh OK root@{ip}:{port}"); return
                time.sleep(10)
            sys.exit("ssh not answering yet")
        time.sleep(10)
    sys.exit("no public IP / port 22 after 10 min")


def cmd_watchdog(a):
    info = read_pod()
    if not info:
        print("no pod to watch"); return
    pid, t0 = info[0], float(info[4])
    while True:
        live = live_ids()
        if live is not None and pid not in live:
            print("pod gone; watchdog exiting"); return
        if (time.time() - t0) / 60 > a.max_minutes:
            print(f"WALL-CLOCK GUARD {a.max_minutes} min - terminating", flush=True)
            cmd_down(a); return
        time.sleep(60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up"); u.add_argument("--disk", type=int, default=30)
    u.add_argument("--max-minutes", type=float, default=45); u.set_defaults(fn=cmd_up)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    d = sub.add_parser("down"); d.add_argument("--pod-id"); d.set_defaults(fn=cmd_down)
    c = sub.add_parser("sshcfg"); c.add_argument("--pod-id"); c.set_defaults(fn=cmd_sshcfg)
    w = sub.add_parser("watchdog"); w.add_argument("--max-minutes", type=float, default=50)
    w.add_argument("--pod-id"); w.set_defaults(fn=cmd_watchdog)
    a = p.parse_args(); a.fn(a)

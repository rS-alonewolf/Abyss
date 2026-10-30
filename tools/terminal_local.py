"""Abyss Terminal on the PC that runs Abyss: the installed run configs (the Models view), the server started detached
- a session of its own, so closing the Terminal leaves it serving - with a PID file beside its config, its stop,
the last --bench summary, and the temperature log the `L` key records.  Standard library only (psutil when there).

The PID file, <config>.pid next to <config>.json: {"pid", "config", "port", "started", "log", "log_pos",
"server_log"} - the pid of the `abyss.py --config <config> --plain` that runs the server (the same start as
./abyss.sh, compile after an update included), its output in <config>.server.log.  A stop checks that the pid is
still that command (never by name: `pgrep -f` matches its own command line).
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
ME = "START-ABYSS.bat" if WIN else "./abyss.sh"
BENCH_FILE = "abyss-bench.txt"
STOP_WAIT_S = 60.0
_CHILDREN: list = []                            # (the detached starts' Popen objects, kept while this process lives)


def read_json(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def list_configs(root: Path = ROOT) -> list:
    """Every installed run config in the Abyss folder (maya-*.json from the setup, abyss-*.json from
    setup-deepseek.sh: an "exe" and its "args"), the most recently used first; old-configs/ is not looked in."""
    out = []
    for p in root.glob("*.json"):
        if p.name.endswith(".shared-settings.json"):
            continue
        c = read_json(p)
        if c.get("exe") and isinstance(c.get("args"), list):
            out.append(p)
    return sorted(out, key=lambda q: q.stat().st_mtime, reverse=True)


def _arg(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def model_label(model_dir):
    """(the model's name with its quant, the quant) from its GGUF set, as the server names it (serve/server.py)."""
    try:
        sys.path.insert(0, str(ROOT))
        from serve.server import model_label as label
        return label(model_dir)
    except Exception:  # noqa: BLE001 - a server that cannot be imported: the config's own names
        return None, None


def config_info(path: Path) -> dict:
    """What the Models view and the header show of a run config, and the files it needs that are missing."""
    path = Path(path)
    c = read_json(path)
    args = [str(a) for a in c.get("args") or []]
    pack = _arg(args, "--glm-pack")
    display, quant = model_label(Path(pack).parent) if pack else (None, None)
    quant = quant or (c.get("installer") or {}).get("quant")
    gpus = c.get("gpu", 0)
    gpus = gpus if isinstance(gpus, list) else [gpus]
    missing = []
    for p, what in ((c.get("exe"), "engine"), (pack, "pack"),
                    (str(Path(c["tokenizer"]) / "vocab.json") if c.get("tokenizer") else None, "tokenizer")):
        if p and not Path(p).exists():
            missing.append(what)
    host = c.get("host") or "127.0.0.1"
    port = int(c.get("port") or 8080)
    keys = str(c.get("api_key") or "")
    return {"path": str(path), "file": path.name, "stem": path.stem, "display": display, "quant": quant,
            "model_name": c.get("model_name"), "ctx": int(_arg(args, "--max-context") or 0) or None,
            "gpus": gpus, "port": port, "host": host, "backend": c.get("backend") or "cuda",
            "log": c.get("log"), "env": c.get("env") or {}, "missing": missing, "has_key": bool(keys),
            "key": keys.split(",")[0] if keys else None, "pack": pack,
            "url": f"http://{'127.0.0.1' if host in ('0.0.0.0', '::', '') else host}:{port}",
            "mtime": path.stat().st_mtime if path.exists() else 0}


def pid_path(cfg: Path) -> Path:
    return Path(cfg).with_suffix(".pid")


def server_log_path(cfg: Path) -> Path:
    return Path(cfg).with_suffix(".server.log")


def cmdline(pid: int) -> list:
    """The process's command line ([] when it is gone or cannot be read)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [x.decode(errors="replace") for x in raw.split(b"\0") if x]
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process(pid).cmdline()
    except Exception:  # noqa: BLE001 - psutil missing, or the process gone
        return []


def alive(pid: int) -> bool:
    if WIN:
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
            return str(pid) in r.stdout
    try:                                       # (a start of this Terminal's that ended: reap it, not a zombie)
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ours(pid: int, cfg: Path) -> bool:
    """Whether `pid` is still the Abyss start of this config (abyss.py or serve/server.py with --config it)."""
    cl = cmdline(pid)
    if not cl:
        return alive(pid) and WIN      # (Windows without psutil: the pid alone)
    text = " ".join(cl)
    return ("abyss.py" in text or "server.py" in text) and Path(cfg).name in text


def read_pid(cfg: Path) -> dict | None:
    """The PID file of a config whose server still runs (a stale file - the process gone or another program now -
    is removed); None: no server of ours."""
    p = pid_path(cfg)
    info = read_json(p)
    pid = info.get("pid")
    if not isinstance(pid, int):
        return None
    if alive(pid) and ours(pid, cfg):
        return info
    try:
        p.unlink()
    except OSError:
        pass
    return None


def running(root: Path = ROOT) -> list:
    """[(config path, PID info)] of every config with a live server of ours."""
    out = []
    for c in list_configs(root):
        info = read_pid(c)
        if info:
            out.append((c, info))
    return out


def start_detached(cfg: Path, python: str | None = None, root: Path = ROOT) -> dict:
    """Start the config's server in a session of its own (Windows: a detached process group), the way ./abyss.sh
    starts it (abyss.py: the engine compiled again after an update, the files checked), without a browser.  Its
    output goes to <config>.server.log; the PID file is written.  -> the PID info."""
    cfg = Path(cfg).resolve()
    info = config_info(cfg)
    log = Path(info["log"]) if info.get("log") else None
    try:
        log_pos = log.stat().st_size if log else 0
    except OSError:
        log_pos = 0
    out_path = server_log_path(cfg)
    cmd = [python or sys.executable, str(root / "abyss.py"), "--config", str(cfg), "--plain"]
    env = dict(os.environ, MAYA_SH="1", ABYSS_NO_BROWSER="1", PYTHONUNBUFFERED="1")
    kw = {}
    if WIN:
        kw["creationflags"] = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    with open(out_path, "wb") as out:
        p = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, close_fds=True, **kw)
    _CHILDREN.append(p)
    pinfo = {"pid": p.pid, "config": str(cfg), "port": info["port"], "started": time.time(),
             "log": str(log) if log else None, "log_pos": log_pos, "server_log": str(out_path)}
    pid_path(cfg).write_text(json.dumps(pinfo, indent=1), encoding="utf-8")
    return pinfo


def signal_stop(pid: int) -> None:
    """Ctrl+C for a detached server: SIGINT to its process group (as a terminal's Ctrl+C reaches the start, the
    server and the engine); Windows: CTRL_BREAK_EVENT to its group."""
    if WIN:
        os.kill(pid, signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(pid, signal.SIGINT)


def force_stop(pid: int) -> None:
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def stop(cfg: Path, link=None, wait: float = STOP_WAIT_S, say=lambda s: None, force: bool = False) -> str:
    """Stop a config's server: POST /api/shutdown when it answers (the request in progress is cut, the engine
    ends), else Ctrl+C to the detached start of ours; after `wait` seconds it is killed.  -> what happened."""
    info = read_pid(cfg)
    asked = False
    if link is not None:
        code, body = link.post("/api/shutdown", {"force": True} if force else {})
        asked = code == 202
        if not asked and info is None:
            msg = (body.get("error") or {}).get("message") if isinstance(body, dict) else None
            return f"the server did not stop: /api/shutdown answered {code or 'nothing'}" + (f" ({msg})" if msg else "")
    if info is None:
        return "stopping: the server ends its engine and exits" if asked else "no server of ours runs for this config"
    pid = info["pid"]
    if not asked:
        try:
            signal_stop(pid)
        except OSError as e:
            return f"could not signal the server ({e})"
    say("waiting for the server to end and free the GPU memory ...")
    t0 = time.time()
    while time.time() - t0 < wait:
        if not alive(pid):
            break
        time.sleep(0.25)
    else:
        force_stop(pid)
        time.sleep(1.0)
    try:
        pid_path(cfg).unlink()
    except OSError:
        pass
    return "stopped" if time.time() - t0 < wait else f"killed after {wait:.0f} s"


def bench_summary(root: Path = ROOT) -> dict | None:
    """The last ./abyss.sh --bench (abyss-bench.txt): its summary line and numbers, and the config it measured."""
    try:
        text = (Path(root) / BENCH_FILE).read_text(encoding="utf-8", errors="replace").splitlines()[:8]
    except OSError:
        return None
    line = next((x for x in text if x.startswith("Abyss v")), None)
    if line is None:
        return None
    out = {"line": line}
    for key, rx in (("decode", r"decode ([\d.]+) tok/s"), ("prompt8k", r"prompt 8k ([\d.]+) tok/s"),
                    ("ssd", r"SSD ([\d.]+) GB/s")):
        m = re.search(rx, line)
        out[key] = float(m.group(1)) if m else None
    m = re.search(r"bottleneck: (\w+)", line)
    out["bottleneck"] = m.group(1) if m else None
    parts = [x.strip() for x in line.split("|")]
    out["model"] = parts[1] if len(parts) > 1 else None
    out["version"] = parts[0].replace("Abyss ", "") if parts else None
    for x in text:
        m = re.search(r"setup: ([^,]+\.json)", x)
        if m:
            out["config"] = m.group(1)
    return out


def version(root: Path = ROOT) -> str | None:
    try:
        return (Path(root) / "VERSION").read_text().strip()
    except OSError:
        return None


class ThermalLog:
    """The `L` key: one CSV row every 2 s into bench/thermal/terminal-<date>.csv (the abyss-thermlog columns' style:
    ISO time first, then °C / W / MHz per device)."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.path: Path | None = None
        self.file = None
        self.writer = None
        self.cols: list = []
        self.last = 0.0

    @property
    def on(self) -> bool:
        return self.file is not None

    def toggle(self) -> Path | None:
        if self.on:
            self.file.close()
            self.file = self.writer = None
            return None
        d = self.root / "bench" / "thermal"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / time.strftime("terminal-%Y%m%d-%H%M%S.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = None
        return self.path

    def row(self, v: dict) -> None:
        """A row from a View (terminal_model.derive), at most every 2 s."""
        if not self.on or time.time() - self.last < 2.0:
            return
        self.last = time.time()
        r = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": v.get("state") or v.get("conn"),
             "tok_s": v["decode"]["now"], "cpu_c": v["cpu"]["temp"], "cpu_w": v["cpu"]["power"],
             "cpu_mhz_avg": round(v["cpu"]["mhz"]) if v["cpu"]["mhz"] else None, "cpu_pct": v["cpu"]["pct"]}
        for c in v["cards"]:
            k = f"g{c.get('index')}_{c.get('role')}"
            r.update({f"{k}_c": c.get("temp"), f"{k}_junction_c": c.get("temp_junction"),
                      f"{k}_mem_c": c.get("temp_mem"), f"{k}_w": c.get("power"), f"{k}_mhz": c.get("clock_mhz"),
                      f"{k}_util": c.get("util")})
        if self.writer is None:
            self.cols = list(r)
            self.writer = csv.DictWriter(self.file, fieldnames=self.cols, extrasaction="ignore")
            self.writer.writeheader()
        self.writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in self.cols})
        self.file.flush()

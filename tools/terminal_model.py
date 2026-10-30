"""Abyss Terminal's numbers: GET /health + GET /metrics (a local or a remote Abyss server) and, on this PC, the engine
log - turned into what the screen shows (derive), with every reading it could not get left as None (drawn as --,
never 0).  Standard library only (no Textual): the screen (abyss_terminal.py), the plain fallback (--plain), --json
and the tests all read the same View.

The parts: Link (one poll: /health, then /metrics with the Bearer key; the connection's state), Poller (Link on a
thread of its own, with back-off), EngineLog (the engine log followed: the per-request "glm stat" lines --bench reads,
the drafter's, the lane, the load's warm-up, warnings and errors), derive (all of it -> a View dict).
"""
from __future__ import annotations

import collections
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import bench_rules as R
from setup_bridge import WARMED, WARMING, LoadWatch

# the connection's states (Link.state) and what the model is doing (View["state"])
CONN = ("connecting", "live", "stale", "disconnected", "auth", "not_abyss", "loading", "stopped", "error")
STALE_S = 10.0                  # a failed poll within this many seconds of the last good one: STALE (values kept)
BACKOFF_MAX_S = 10.0
PHASE2 = "engine counter (phase 2)"   # the hint under a tile whose number the engine does not report yet
LOCAL_ONLY = "local only"             # ... and under one that needs the engine log on this PC
_CTRL = re.compile("[\x00-\x1f\x7f-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]")
QUANT_RE = re.compile(r"(?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|MXFP4|BF16|F16|F32)(?![A-Za-z0-9])")
# the engine log's lines the Terminal reads besides --bench's (bench_rules): a start, the lane, the mirror, trouble
START_LINE = re.compile(r"VRAM before the expert pool")
LANE_LINE = re.compile(r"iGPU lane (on|off|failed|stall)|CPU lane off|CPU lane: (\d+) threads")
MIRROR_LINE = re.compile(r"glm fast mirror \S+: ([\d.]+) GB from the shards' drive, ([\d.]+) GB from the mirror")
KIND = (("stat", re.compile(r"glm stat:")), ("spec", re.compile(r"glm spec:")),
        ("watchdog", re.compile(r"watchdog", re.I)), ("error", re.compile(r"\bERR\b|error|failed|out of memory|"
                                                                          r"Traceback", re.I)),
        ("warn", re.compile(r"WARNING|warning")), ("note", re.compile(r"^NOTE|\bNOTE\b")),
        ("load", re.compile(r"warming the expert tiers|tiers warm|RAM tier|pinned")))


def clean(s, limit: int = 200) -> str:
    """A string from a server or a log, safe to draw: no control or bidi characters, at most `limit` long."""
    return _CTRL.sub("", str(s))[:limit] if s is not None else ""


def num(v):
    """A number from /metrics (the engine's INFO facts can come as text) - else None.  True/False are not numbers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def kind_of(line: str) -> str:
    """The log line's kind for the Logs view's colours."""
    for k, rx in KIND:
        if rx.search(line):
            return k
    return "info"


def pretty_name(display_name, quant, slug=None, config=None) -> tuple:
    """("DeepSeek V4.1 Flash 1017", "UD-IQ2_XXS") from the server's display name ("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS")
    - the quant split off, hyphens to spaces; the config's model_name slug when there is no display name."""
    name = clean(display_name or slug or config or "", 80)
    q = clean(quant or "", 24) or None
    if q is None:
        found = QUANT_RE.findall(name)
        q = found[-1] if found else None
    if q and q in name:                            # (the quant and what follows it: "IQ3_XXS-B", "UD-Q4_K_XL")
        at = name.rindex(q)
        name, q = name[:at].rstrip("-_ ."), name[at:].strip("-_ ") or q
    if display_name:
        name = name.replace("-", " ")
    return name or "unknown model", q


def badges(engine: dict, env: dict | None = None) -> list:
    """The settings that change the answers or how the model runs, for the header: [(label, tone)] - tone "alert"
    for the ones that change the output (uncensor, tail skip, defer, abliteration).  `env`: the config's (local)."""
    env = {k: str(v) for k, v in (env or {}).items()}
    out = []
    unc = engine.get("uncensor")
    if (unc is not None and str(unc) not in ("0", "", "None", "False")) or \
            str(engine.get("cvec", 0)) not in ("0", "", "None"):
        out.append(("UNCENSOR", "alert"))
    skip = engine.get("tail_skip") or env.get("STRATA_GLM_TAIL_SKIP")
    if skip and str(skip) != "0":
        out.append((f"TAIL SKIP {skip}", "alert"))
    if str(engine.get("defer") or env.get("STRATA_GLM_DEFER") or "0") not in ("0", ""):
        out.append(("DEFER", "alert"))
    if str(env.get("STRATA_GLM_ABL") or "0") not in ("0", ""):
        out.append(("ABL", "alert"))
    drafter = engine.get("drafter")
    if drafter and drafter != "none":
        out.append((str(drafter).upper(), "info"))
    elif str(engine.get("mtp") or "0") not in ("0", ""):
        out.append(("MTP", "info"))
    if engine.get("mirror") or env.get("STRATA_GLM_MIRROR"):
        out.append(("MIRROR", "info"))
    if engine.get("helper") or str(env.get("STRATA_GLM_EXPERT_HELPER") or "0") not in ("0", ""):
        out.append(("HELPER", "info"))
    return out


# ------------------------------------------------------------------------------------------------ the server
class Link:
    """One Abyss server: poll() reads /health (no key needed) and /metrics (the Bearer key, when given) and sets the
    connection's state.  `opener(request, timeout)`: urllib's by default (tests hand their own)."""

    def __init__(self, base: str, key: str | None = None, timeout: float = 3.0, opener=None):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.health: dict | None = None
        self.metrics: dict | None = None
        self.state = "connecting"
        self.error = ""
        self.last_ok: float | None = None          # time.time() of the last good /metrics
        self.fails = 0
        self.ever_ok = False
        self.instance = None                       # the server's start (totals.since): a new one resets the history

    def _get(self, path: str, auth: bool):
        h = {"Accept": "application/json"}
        if auth and self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, headers=h)
        with self.opener(req, timeout=self.timeout) as r:
            raw = r.read(4 * 2**20)                # (2 MiB is far over /metrics' size: a cap, not a limit)
        return json.loads(raw.decode("utf-8", "replace"))

    def post(self, path: str, body: dict) -> tuple:
        """(HTTP status, JSON answer or {}) of a POST - /api/shutdown."""
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=h, method="POST")
        try:
            with self.opener(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            with e:
                try:
                    return e.code, json.loads(e.read().decode() or "{}")
                except ValueError:
                    return e.code, {}
        except (OSError, ValueError) as e:
            return 0, {"error": {"message": str(e)}}

    def poll(self, all_requests: bool = False) -> str:
        try:
            health = self._get("/health", auth=False)
        except urllib.error.HTTPError as e:
            e.close()
            return self._failed(f"/health answered {e.code}")
        except (OSError, ValueError) as e:
            return self._failed(getattr(e, "reason", None) or str(e))
        self.health = health if isinstance(health, dict) else {}
        try:
            m = self._get("/metrics?requests=all" if all_requests else "/metrics", auth=True)
        except urllib.error.HTTPError as e:
            e.close()
            if e.code in (401, 403):
                self.state, self.error, self.fails = "auth", f"/metrics answered {e.code}", 0
                return self.state
            if self.health.get("service") not in ("abyss", None):
                self.state, self.error = "not_abyss", f"/metrics answered {e.code}"
                return self.state
            return self._failed(f"/metrics answered {e.code}")
        except (OSError, ValueError) as e:
            return self._failed(getattr(e, "reason", None) or str(e))
        if not isinstance(m, dict):
            return self._failed("/metrics is not a JSON object")
        since = (m.get("totals") or {}).get("since")
        if self.instance is not None and since is not None and since != self.instance:
            self.metrics = None                    # (another start of the server: nothing carried over)
        self.instance = since
        self.metrics = m
        self.last_ok, self.fails, self.ever_ok, self.error = time.time(), 0, True, ""
        self.state = "live" if self.health.get("service", "abyss") == "abyss" else "not_abyss"
        return self.state

    def _failed(self, why) -> str:
        self.fails += 1
        self.error = clean(why, 160)
        if self.ever_ok and self.last_ok is not None and time.time() - self.last_ok < STALE_S and self.fails < 3:
            self.state = "stale"
        else:
            self.state = "disconnected"
        return self.state

    def next_delay(self, interval: float) -> float:
        """Seconds to the next poll: the interval while it answers; after failures 1, 2, 4 ... up to 10 s, with a
        little jitter (several terminals on one server do not poll in step)."""
        if self.state == "auth":
            return float("inf")                    # (stopped: `r` tries again)
        if self.fails == 0:
            return interval
        return min(BACKOFF_MAX_S, interval * 2 ** (self.fails - 1)) * random.uniform(0.85, 1.15)


class Poller:
    """A Link polled on a thread of its own; `kick()` polls now (the `r` key), `all_requests` asks for every kept
    request (the Requests view).  The screen reads `link` under `lock`."""

    def __init__(self, link: Link, interval: float = 1.0):
        self.link, self.interval = link, interval
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.all_requests = False
        self.paused = False
        self.polls = 0
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="abyss-terminal-poll")

    def start(self) -> "Poller":
        self.thread.start()
        return self

    def kick(self) -> None:
        if self.link.state == "auth":
            self.link.state = "connecting"
        self.wake.set()

    def stop(self) -> None:
        self._stop = True
        self.wake.set()

    def _run(self) -> None:
        while not self._stop:
            with self.lock:
                self.link.poll(self.all_requests)
                self.polls += 1
            d = self.link.next_delay(self.interval)
            self.wake.wait(None if d == float("inf") else d)
            self.wake.clear()


# ------------------------------------------------------------------------------------------------ the engine log
class EngineLog:
    """The engine log on this PC, followed from where it stands (the first read looks back over the last start, so
    the lane and the last requests' "glm stat" lines are known at once).  `recs`: --bench's records (bench_rules)
    with `t` = when the line was read (None for lines from before the Terminal opened)."""

    FIRST_LOOK_BYTES = 32 * 2**20
    STEP_BYTES = 256 * 2**10

    def __init__(self, path, gpus=(0,), vram=None, keep: int = 2000, from_end: bool = False, start_at=None):
        self.path = Path(path) if path else None
        self.lines: collections.deque = collections.deque(maxlen=keep)   # (kind, text)
        self.recs: list = []
        self.lane: str | None = None               # "iGPU lane" / "CPU lane" / None (not seen)
        self.lane_state: str | None = None         # on / off / failed / stall
        self.lane_threads: int | None = None
        self.mirror: tuple | None = None           # (GB from the shards' drive, GB from the mirror) at the engine's end
        self.last_error: tuple | None = None       # (time read, text)
        self.last_warning: str | None = None
        self.watchdog: tuple | None = None
        self.last_line = ""
        self.load = LoadWatch(str(self.path) if self.path else None, gpus, vram)
        self.started = False                       # a start line seen
        self.pos = 0
        self.first = True
        if start_at is not None:                   # (a start of ours: its lines only, from where the log stood)
            self.pos, self.first = int(start_at), False
        elif from_end and self.path:
            try:
                self.pos = self.path.stat().st_size
                self.first = False
            except OSError:
                pass

    def read(self) -> int:
        """The log's new whole lines (at most STEP_BYTES a call after the first): how many."""
        if self.path is None:
            return 0
        try:
            size = self.path.stat().st_size
            if size < self.pos:                    # (rotated / truncated: start again)
                self.pos, self.first = 0, True
            with open(self.path, "rb") as f:
                if self.first:
                    self.pos = max(0, size - self.FIRST_LOOK_BYTES)
                f.seek(self.pos)
                chunk = f.read(self.FIRST_LOOK_BYTES if self.first else self.STEP_BYTES)
        except OSError:
            return 0
        cut = chunk.rfind(b"\n") + 1
        self.pos += cut
        lines = chunk[:cut].decode("utf-8", "replace").splitlines()
        if self.first:                             # (the look back: from the last start on, kept short)
            starts = [i for i, x in enumerate(lines) if START_LINE.search(x)]
            older, lines = lines[:starts[-1]] if starts else [], lines[starts[-1]:] if starts else lines
            for x in older:                         # (the requests before the last start still count)
                self._stat(x, None)
        stamp = None if self.first else time.time()
        for x in lines:
            self.feed(x, stamp)
        self.first = False
        return len(lines)

    def _stat(self, x: str, stamp) -> bool:
        m = R.STAT_LINE.search(x)
        if m:
            rec = {k: float(v) for k, v in m.groupdict().items()}
            rec["t"] = stamp
            self.recs.append(rec)
            del self.recs[:-200]
            return True
        if self.recs:
            m = R.SPEC_LINE.search(x)
            if m and "spec" not in self.recs[-1]:
                self.recs[-1]["spec"] = (int(m.group(1)), int(m.group(2)))
            m = R.GAP_LINE.search(x)
            if m:
                self.recs[-1]["gpu_busy"], self.recs[-1]["gpu_idle"] = float(m.group("busy")), float(m.group("idle"))
        return False

    def feed(self, x: str, stamp=None) -> None:
        x = clean(x.rstrip("\r\n"), 400)
        if not x.strip():
            return
        k = kind_of(x)
        self.lines.append((k, x))
        self.last_line = x
        if START_LINE.search(x):                   # a new start of the engine
            self.started = True
            self.lane = self.lane_state = self.lane_threads = None
            self.mirror = None
            self.load.reset()
        self._stat(x, stamp)
        if R.IGPU_LANE in x:
            self.lane = "iGPU lane"
        m = LANE_LINE.search(x)
        if m:
            if m.group(1):
                self.lane_state = m.group(1)
                if m.group(1) in ("off", "failed"):
                    self.lane = "CPU lane"
            elif m.group(2):
                self.lane_threads = int(m.group(2))
                self.lane = self.lane or "CPU lane"
                self.lane_state = self.lane_state or "on"
            else:
                self.lane, self.lane_state = "CPU lane", "off"
        if m := MIRROR_LINE.search(x):
            self.mirror = (float(m.group(1)), float(m.group(2)))
        if m := WARMING.search(x):
            self.load.parts[int(m.group(2))] = [float(m.group(3)), float(m.group(4))]
        elif m := WARMED.search(x):
            self.load.parts[int(m.group(2))] = [float(m.group(3))] * 2
        if k == "error":
            self.last_error = (stamp, x)
        elif k == "warn":
            self.last_warning = x
        elif k == "watchdog":
            self.watchdog = (stamp, x)

    def decode_recs(self, n: int = 4) -> list:
        """The last `n` records that wrote an answer (a prompt-only request has no decode: under 1 ms a token)."""
        return [r for r in self.recs if r.get("ms", 0) >= 1.0][-n:]

    def match(self, request: dict) -> dict | None:
        """The "glm stat" record of a request (/metrics requests[]): the one read within a few seconds of its end -
        lines from before the Terminal opened carry no time and are not matched."""
        end = (num(request.get("time")) or 0) + (num(request.get("duration_s")) or 0)
        best = None
        for r in self.recs:
            if r.get("t") is not None and abs(r["t"] - end) <= 4.0:
                if best is None or abs(r["t"] - end) < abs(best["t"] - end):
                    best = r
        return best


# ------------------------------------------------------------------------------------------------ the View
def _last(xs):
    for x in reversed(xs or []):
        if x is not None:
            return x
    return None


def token_time(m: dict, log: EngineLog | None, backend: str | None) -> dict | None:
    """Where a token's time goes: the engine's live STAT when it carries the phase-2 counters (disk_ms, lane_ms),
    else the engine log's last answers on this PC (--bench's records), else None.  The verdict and its advice come
    from bench_rules - the same as --bench's."""
    tn = ((m or {}).get("tiers") or {}).get("now") or {}
    lane_name = (log.lane if log and log.lane else None) or "CPU lane"
    if num(tn.get("disk_ms")) is not None and num(tn.get("lane_ms")) is not None and num(tn.get("ms_tok")):
        rec = {"ms": num(tn["ms_tok"]), "disk_ms": num(tn["disk_ms"]), "lane_ms": num(tn["lane_ms"]),
               "ram_fetch": num(tn.get("ram_fetch")) or 0.0, "vram_hit": 100 * (num(tn.get("vram_hit")) or 0),
               "disk_reads": num(tn.get("disk")) or 0.0, "lane": num(tn.get("lane")) or 0.0,
               "tail_skip": num(tn.get("tail_skip")) or 0.0, "disk_waits": num(tn.get("disk_waits")) or 0.0}
        recs, source = [rec], "live"
    elif log is not None and log.decode_recs():
        recs, source = log.decode_recs(), "log"
    else:
        return None
    verdict, evidence = R.bench_verdict(recs, lane_name)
    mean = lambda k: R._mean(r.get(k, 0.0) for r in recs)   # noqa: E731
    what, moves = R.advice(verdict, backend)
    spec = [r["spec"] for r in recs if "spec" in r]
    return {"source": source, "n": len(recs), "ms": mean("ms"), "disk_ms": mean("disk_ms"),
            "lane_ms": mean("lane_ms"), "ram_fetch": mean("ram_fetch"), "vram_hit": mean("vram_hit"),
            "disk_reads": mean("disk_reads"), "disk_waits": mean("disk_waits"), "lane": mean("lane"),
            "tail_skip": mean("tail_skip"), "lane_name": lane_name, "verdict": verdict, "evidence": evidence,
            "what": what, "moves": moves,
            "spec": (sum(a for a, _ in spec), sum(b for _, b in spec)) if spec else None}


def cards(m: dict) -> list:
    """Every card /metrics lists (hardware.gpus, one card too since the phase-1.5 server), else the totals as one."""
    hw, st = (m or {}).get("hardware") or {}, (m or {}).get("hardware_static") or {}
    out = []
    names = {g.get("index"): g for g in st.get("gpus") or []}
    for g in hw.get("gpus") or []:
        c = dict(g)
        c["name"] = clean(c.get("name") or (names.get(g.get("index")) or {}).get("name") or "", 60)
        c["role"] = c.get("role") or "main"
        out.append(c)
    if not out and (hw.get("gpu_mem_total") is not None or hw.get("gpu_util") is not None):
        c = {k[4:]: v for k, v in hw.items() if k.startswith("gpu_")}
        c.update(index=0, role="main", name=clean(st.get("gpu_name") or "", 60))
        out.append(c)
    return out


def short_gpu(name: str) -> str:
    for a in ("NVIDIA ", "GeForce ", "AMD ", "Radeon ", "(TM)", "Graphics"):
        name = name.replace(a, "")
    return " ".join(name.split())


def derive(link: Link | None, local: dict | None = None, log: EngineLog | None = None, now: float | None = None,
           seen: dict | None = None) -> dict:
    """Everything the screen shows, from the last poll (`link`), this PC's facts (`local`: the config's info, whether
    a server process of ours runs, the bench file - terminal_local) and the engine log.  None = unknown.  `seen`: what the
    caller keeps between frames (the best PCIe link seen while working)."""
    now = time.time() if now is None else now
    local = local or {}
    m = (link.metrics if link else None) or {}
    health = (link.health if link else None) or {}
    conn = link.state if link else "disconnected"
    eng = m.get("engine") or {}
    live = m.get("live") or {}
    hw = m.get("hardware") or {}
    st = m.get("hardware_static") or {}
    hist = m.get("history") or {}
    tiers = m.get("tiers") or {}
    tn = tiers.get("now") or {}
    reqs = m.get("requests") or []
    last = reqs[0] if reqs else None
    cfg = local.get("config") or {}
    v: dict = {"conn": conn, "error": clean(link.error) if link else "", "remote": bool(local.get("remote")),
               "url": link.base if link else None, "age": (now - link.last_ok) if link and link.last_ok else None,
               "has_metrics": bool(m), "_keyed": bool(link and link.key)}

    # what this server is: the name with its quant, the config, the context, the backend, the versions, badges
    name, quant = pretty_name(eng.get("display_name") or cfg.get("display"), eng.get("quant") or cfg.get("quant"),
                              eng.get("model") or health.get("model") or cfg.get("model_name"),
                              eng.get("config") or cfg.get("stem"))
    v.update(name=name, quant=quant, config=clean(eng.get("config") or cfg.get("file") or "", 60) or None,
             ctx=num(eng.get("max_context")) or num(health.get("max_context")) or num(cfg.get("ctx")),
             backend=clean(eng.get("backend") or cfg.get("backend") or "", 8).upper() or None,
             version=clean(eng.get("maya_version") or health.get("version") or local.get("version") or "", 20) or None,
             engine_version=clean(eng.get("version") or "", 20) or None,
             badges=badges(eng, cfg.get("env")), service=clean(health.get("service") or "", 20) or None)

    # the connection and what the model does
    pid_alive = local.get("pid_alive")       # (None: not known - a remote server, or one without our PID file)
    if conn == "disconnected":
        if pid_alive:
            conn = "loading"                       # (the server answers only once its engine is ready)
        elif pid_alive is False:
            conn = "stopped"
    if log is not None and log.last_error and log.last_error[0] and now - log.last_error[0] < 30 and \
            conn in ("disconnected", "stopped"):
        conn = "error"
    v["conn"] = conn
    state = live.get("state") if m else None
    hs = health.get("status")
    if hs in ("reloading", "updating"):
        state = hs
    elif state == "idle" and live.get("queued"):
        state = "queued"
    v["state"] = state
    v["queued"] = live.get("queued") or 0
    v["phase"] = clean(live.get("phase") or "", 20) or None
    v["reload"] = m.get("reload")

    # decode (live, else the last request's) and prompt (live while reading, else the last request's)
    gen = state == "generating"
    reading = state == "reading"
    v["decode"] = {"now": num(live.get("tok_s")) if gen else None,
                   "mean": num(live.get("tok_s_mean")) if gen else None,
                   "last": num(last.get("decode_tok_s")) if last else None,
                   "ms": num(tn.get("ms_tok")) if gen and tn else None,
                   "spark": list(hist.get("tok_s") or []),
                   "generated": live.get("generated") if gen else None, "max_tokens": live.get("max_tokens")}
    d = v["decode"]
    d["value"] = d["now"] if d["now"] is not None else d["last"]
    d["label"] = "live" if d["now"] is not None else "last request" if d["last"] is not None else None
    if d["ms"] is None and d["value"]:
        d["ms"] = 1000.0 / d["value"]
    last_prefill = None
    if last:
        last_prefill = num(last.get("prefill_tok_s"))
        if last_prefill is None:
            read = (num(last.get("prompt_tokens")) or 0) - (num(last.get("reused")) or 0)
            pm = num(last.get("prompt_ms")) or 0
            last_prefill = read / (pm / 1000) if pm > 0 and read > 0 else None
    v["prompt"] = {"now": num(live.get("prefill_tok_s_mean")) if reading else None, "last": last_prefill,
                   "read": live.get("prompt_read"), "total": live.get("prompt_total") or live.get("prompt_tokens"),
                   "spark": list(hist.get("prefill_tok_s_mean") or [])}
    p = v["prompt"]
    p["value"] = p["now"] if p["now"] is not None else p["last"]
    p["label"] = "live" if p["now"] is not None else "last request" if p["last"] is not None else None
    v["elapsed"] = num(live.get("elapsed_s"))

    # where a token's time goes, and the bottleneck (--bench's rules)
    v["tt"] = token_time(m, log, (eng.get("backend") or cfg.get("backend") or "").lower() or None)
    v["tt_why"] = None if v["tt"] else (PHASE2 + " - remote: until then the local engine log has it"
                                        if local.get("remote") else "no answer written yet in this engine log"
                                        if log is not None else PHASE2)

    # the expert tiers (STAT, the fast path) and the context
    v["tiers"] = {k: num(tn.get(k)) for k in ("vram_hit", "ram_fetch", "disk", "promo", "vram_used", "vram_slots",
                                               "ram_used", "ram_slots", "vram_gb", "ram_gb", "tok_s", "ms_tok")}
    v["tiers"]["age"] = (now - tn["time"]) if tn.get("time") else None
    v["tiers"]["history"] = tiers.get("history") or {}
    for k, ik in (("vram_slots", "vram_slots"), ("ram_slots", "ram_slots"), ("vram_gb", "vram_gb"), ("ram_gb", "ram_gb")):
        if v["tiers"][k] is None:
            v["tiers"][k] = num(eng.get(ik))
    v["experts"] = num(eng.get("experts"))
    if state in ("reading", "generating"):
        used = (num(live.get("prompt_tokens")) or 0) + (num(live.get("generated")) or 0)
        reused = None
    elif last:
        used = (num(last.get("prompt_tokens")) or 0) + (num(last.get("output_tokens")) or 0)
        reused = num(last.get("reused"))
    else:
        used, reused = None, None
    v["context"] = {"used": used, "reused": reused, "max": v["ctx"], "live": state in ("reading", "generating"),
                    "kv_gb": num(eng.get("kv_gb")), "kv_ctx": num(eng.get("kv_ctx"))}

    # the cards: the model's (main / split), a helper, the iGPU lane - the busiest and hottest of the model's own
    cs = cards(m)
    model_cards = [c for c in cs if c.get("role") in ("main", "split")] or [c for c in cs if c.get("role") != "igpu"]
    loads = [num(c.get("util")) for c in model_cards if num(c.get("util")) is not None]
    temps = [(num(c.get("temp")), num(c.get("temp_junction"))) for c in model_cards if num(c.get("temp")) is not None]
    hot = max(temps, default=(None, None), key=lambda t: t[0])
    v["cards"] = cs
    v["gpu"] = {"busiest": max(loads) if loads else num(hw.get("gpu_util")), "n": len(model_cards),
                "mem_used": num(hw.get("gpu_mem_used")), "mem_total": num(hw.get("gpu_mem_total")),
                "temp": hot[0] if hot[0] is not None else num(hw.get("gpu_temp")),
                "junction": hot[1] if hot[0] is not None else num(hw.get("gpu_temp_junction")),
                "power": num(hw.get("gpu_power")), "power_limit": num(hw.get("gpu_power_limit")),
                "spark": list(hist.get("gpu_util") or []), "pool_gb": num(eng.get("vram_gb"))}
    # PCIe: the narrowest working link of the model's cards, against what they can do; a drop counts only while the
    # model works (an idle card drops to Gen1 by itself, as --bench's notes say)
    links = [(num(c.get("pcie_gen")), num(c.get("pcie_width")), num(c.get("pcie_gen_max")),
              num(c.get("pcie_width_max")), num(c.get("pcie_rx_mb"))) for c in model_cards]
    links = [x for x in links if x[0] is not None]
    if not links and num(hw.get("gpu_pcie_gen")) is not None:
        links = [(num(hw.get("gpu_pcie_gen")), num(hw.get("gpu_pcie_width")), num(hw.get("gpu_pcie_gen_max")),
                  num(hw.get("gpu_pcie_width_max")), num(hw.get("gpu_pcie_rx_mb")))]
    if links:
        worst = min(links, key=lambda x: (x[0] or 0) * (x[1] or 1))
        rx = [x[4] for x in links if x[4] is not None]
        v["pcie"] = {"gen": worst[0], "width": worst[1], "gen_max": worst[2], "width_max": worst[3],
                     "rx_mb": sum(rx) if rx else None}
        working = state in ("reading", "generating")
        # a drop: while working, narrower than the best link seen while working in this session (an NVIDIA card's
        # "max" is the card's own - x16 on an x4 OCuLink cable - so it is shown, not judged)
        here = (worst[0] or 0) * (worst[1] or 1)
        best = seen.get("pcie_best") if seen is not None else None
        if working and seen is not None and (best is None or here > best[0]):
            seen["pcie_best"] = best = (here, worst[0], worst[1])
        v["pcie"]["working"] = working
        v["pcie"]["dropped"] = bool(working and best and here < best[0])
        v["pcie"]["best"] = best[1:] if best else None
        v["pcie"]["below_max"] = bool((worst[2] and worst[0] < worst[2]) or
                                      (worst[3] and worst[1] and worst[1] < worst[3]))
    else:
        v["pcie"] = None

    # the CPU, RAM, swap and pressure, the disks
    mhz = [x for x in (hw.get("cpu_mhz") or []) if x]
    v["cpu"] = {"pct": num(hw.get("cpu")), "cores": st.get("cores"), "threads": st.get("threads"),
                "mhz": sum(mhz) / len(mhz) if mhz else None, "mhz_max": max(mhz) if mhz else None,
                "temp": num(hw.get("cpu_temp")), "power": num(hw.get("cpu_power")),
                "power_blocked": hw.get("cpu_power_blocked"), "isa": st.get("cpu_isa"),
                "name": clean(st.get("cpu_name") or st.get("cpu_model") or "", 60) or None,
                "per_core": hw.get("cpu_cores") or [], "clusters": hw.get("cpu_clusters") or [],
                "governor": st.get("cpu_governor"), "epp": st.get("cpu_epp"),
                "lane_threads": num(eng.get("cpu_threads")) or (log.lane_threads if log else None),
                "throttling": hw.get("cpu_throttling"), "spark": list(hist.get("cpu") or [])}
    v["ram"] = {"used": num(hw.get("ram_used")), "total": num(hw.get("ram_total")),
                "cached": num(hw.get("ram_cached")), "swap_used": num(hw.get("swap_used")),
                "swap_total": num(hw.get("swap_total")), "psi_some": num(hw.get("psi_mem_some")),
                "psi_full": num(hw.get("psi_mem_full")), "spark": list(hist.get("ram_used") or [])}
    drives = []
    sdrives = {d.get("name"): d for d in st.get("drives") or []}
    for dv in hw.get("drives") or []:
        drives.append({"name": clean(dv.get("name") or "?", 30), "roles": list(dv.get("roles") or []),
                       "read_mb": num(dv.get("read_mb")), "write_mb": num(dv.get("write_mb")),
                       "paths": (sdrives.get(dv.get("name")) or {}).get("paths") or []})
    model_mb = sum(d["read_mb"] or 0 for d in drives if "model" in d["roles"])
    mirror_mb = sum(d["read_mb"] or 0 for d in drives if "mirror" in d["roles"] and "model" not in d["roles"])
    v["disk"] = {"read_mb": num(hw.get("disk_read_mb")), "write_mb": num(hw.get("disk_write_mb")),
                 "drives": drives, "spark": list(hist.get("disk_read_mb") or []),
                 "mirror_share": mirror_mb / (model_mb + mirror_mb) if mirror_mb + model_mb > 1.0 and
                 any("mirror" in d["roles"] for d in drives) else None,
                 "mirror_end": log.mirror if log else None,
                 "free": num((m.get("storage") or {}).get("free")), "total": num((m.get("storage") or {}).get("total"))}

    # the drafter: the last request's and the server's totals (drafts accepted / offered)
    tot = m.get("totals") or {}
    da, do = (num(last.get("drafts_accepted")), num(last.get("drafts_offered"))) if last else (None, None)
    ta, to = num(tot.get("drafts_accepted")), num(tot.get("drafts_offered"))
    v["drafter"] = {"last": da / do if da is not None and do else None, "total": ta / to if ta is not None and to
                    else None, "accepted": ta, "offered": to, "kind": clean(eng.get("drafter") or "", 12) or
                    ("mtp" if str(eng.get("mtp") or "0") not in ("0", "") else None)}
    v["requests"], v["totals"] = reqs, tot
    v["lane"] = {"name": log.lane if log else None, "state": log.lane_state if log else None,
                 "share": num(eng.get("pcie_share"))}

    # the start, on this PC: how far the warm-up is (the engine log), its stage and time
    if conn == "loading" and log is not None:
        frac = None
        if log.load.parts:
            total = log.load.total()[0]
            frac = min(1.0, sum(done for done, _ in log.load.parts.values()) / max(total, 1e-9))
        v["loading"] = {"frac": frac, "stage": log.last_line, "elapsed": now - local["started"]
                        if local.get("started") else None}
    else:
        v["loading"] = None
    v["bench"] = local.get("bench")
    v["banners"] = banners(v, log, now)
    return v


def banners(v: dict, log: EngineLog | None, now: float) -> list:
    """The one-line warnings over the view: [(text, tone)]."""
    out = []
    g = v["gpu"]
    hot = max((x for x in (g.get("junction"), g.get("temp")) if x is not None), default=None)
    throttle = sorted({t for c in v["cards"] for t in (c.get("throttle") or []) if t not in ("idle", "gpu_idle")})
    if (g.get("junction") or 0) >= 100 or (g.get("temp") or 0) >= 87:
        out.append((f"GPU HOT {hot:.0f}°C - it may slow itself down" + (f" ({', '.join(throttle)})" if throttle
                                                                         else ""), "alert"))
    elif throttle and v.get("state") in ("generating", "reading"):
        out.append((f"GPU throttling: {', '.join(throttle)}", "warn"))
    p = v.get("pcie")
    if p and p.get("dropped"):
        b = p.get("best") or (p["gen"], p["width"])
        out.append((f"PCIe link fell to Gen{p['gen']:.0f} x{(p['width'] or 0):.0f} while working (it ran Gen{b[0]:.0f} "
                    f"x{(b[1] or 0):.0f} earlier) - check the cable / USB4 / OCuLink", "alert"))
    r = v["ram"]
    avail = r["total"] - r["used"] if r.get("total") and r.get("used") is not None else None
    if (r.get("psi_full") or 0) >= 5 or (avail is not None and avail < 2 * 2**30):
        out.append(("MEMORY PRESSURE: the system may stop the server (oomd) - lower STRATA_GLM_RAM_GB / close apps",
                    "alert"))
    elif (r.get("psi_some") or 0) >= 10 or (r.get("swap_used") or 0) > 2 * 2**30 or \
            (avail is not None and avail < 4 * 2**30):
        out.append(("memory is tight: swapping or waiting for RAM (PSI some "
                    f"{(r.get('psi_some') or 0):.0f}%, swap {(r.get('swap_used') or 0) / 2**30:.1f} GB)", "warn"))
    if v["conn"] == "stale":
        out.append((f"no answer from the server for {v['age']:.0f} s - the numbers are old", "warn"))
    if v["conn"] == "auth":
        out.append(("AUTH REQUIRED: /metrics wants the API key - --key-env NAME (the variable that holds it), "
                    "then r", "alert"))
    if v["remote"] and v["url"] and v["url"].startswith("http://") and \
            not re.match(r"http://(127\.|localhost|\[::1\])", v["url"]) and v.get("_keyed"):
        out.append(("plain http: the key travels unencrypted - Tailscale, or ssh -L 8082:127.0.0.1:8082", "warn"))
    if log is not None:
        if log.watchdog and log.watchdog[0] and now - log.watchdog[0] < 300:
            out.append(("watchdog: " + log.watchdog[1][:120], "alert"))
        if log.lane_state in ("failed", "stall"):
            out.append((f"iGPU lane {log.lane_state} - the CPU computes the lane now", "warn"))
    tt = v.get("tt")
    if tt and tt["verdict"] == "ssd" and tt["source"] == "live":
        out.append(("SSD-bound: " + tt["what"], "warn"))
    return out

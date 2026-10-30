#!/usr/bin/env python3
"""Abyss Terminal: a terminal UI for an Abyss server - this PC's or another's - that you open and close without
stopping the model.  It reads the server's GET /health and GET /metrics once a second and, on this PC, the engine log
(where a token's time goes, with --bench's verdict; the load's progress), lists the installed run configs, starts a
server detached (it keeps serving after the Terminal closes) and stops it.

    ./abyss.sh --terminal                          the last-used config's server: attach, or offer to start it
    ./abyss.sh --terminal --config abyss-x.json    that config's
    ./abyss.sh --terminal --url http://host:8082 --key-env ABYSS_API_KEY     a remote server (read-only)
    ./abyss.sh --terminal --plain                  no Textual: plain text redrawn every second (old consoles)
    ./abyss.sh --terminal --once / --json          one reading, printed (scripts, issue reports)
    ./abyss.sh --terminal --demo r9700             recorded readings, no server needed (r9700 lowspec split loading
                                                   down)
    python tools/abyss_terminal.py ...             the same without abyss.sh (python -m tools.abyss_terminal too)

The screen itself is terminal_app.py (Textual, from third_party/wheels); the numbers terminal_model.py, the drawing
terminal_draw.py, this PC's configs and processes terminal_local.py - those three need only the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import terminal_draw as D  # noqa: E402
import terminal_local as T  # noqa: E402
import terminal_model as M  # noqa: E402

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")


class Session:
    """What the Terminal is attached to and what it shows: the server (Link + Poller), this PC's config, PID file and
    engine log, and the screen's state (view, selection, freeze ...).  Both the Textual screen and --plain drive it."""

    def __init__(self, url: str | None = None, key_env: str | None = None, config: Path | None = None,
                 interval: float = 1.0, demo: str | None = None, root: Path = T.ROOT, poll: bool = True):
        self.root = Path(root)
        self.interval = max(0.5, min(10.0, interval))
        self.demo = demo
        self.remote = bool(url) and not demo
        self.key_env = key_env
        self.started_by_us = False
        self.seen: dict = {}
        self.view = 0
        self.sel = {3: 0, 7: 0}
        self.help = False
        self.frozen = False
        self.frozen_view = None
        self.theme = "mono" if os.environ.get("NO_COLOR") else "dark"
        self.message, self.message_until = "", 0.0
        self.filter = ""
        self.log_scroll = 0
        self.check = ""
        self.tick_n = 0
        self.thermal = T.ThermalLog(self.root)
        self.busy = None                           # a start / stop / bench running on a thread: what it does
        self.bench_log: M.EngineLog | None = None
        self.cfg_path: Path | None = None
        self.cfg: dict | None = None
        self.configs: list | None = None
        self.log: M.EngineLog | None = None
        self.pid_info = None
        self._last_pid_check = self._last_cfg_scan = 0.0
        self.poll = poll
        self.fake = None
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key:
            self.say(f"${key_env} is empty: no key sent", 8)
        if demo:
            import terminal_fixtures as F
            self.fake = F.FakeServer(demo if demo in F.SCENARIOS else "r9700")
            if demo in ("loading", "down"):
                self.fake.down()
            base = self.fake.url
            self.demo_local = F.local(demo, now=time.time())
            self.configs = [dict(c) for c in F.CONFIGS]
            if demo in F.LOGS:
                self.log = M.EngineLog(None)
                for x in F.LOGS[demo]:
                    self.log.feed(x, None)
            self.remote = demo == "split"
            if self.remote:
                self.configs = None
        elif url:
            base = url.rstrip("/")
            if "://" not in base:
                base = "http://" + base
        else:
            base = self._pick_local(config)
        self.link = M.Link(base, key if (url or demo) else (self.cfg or {}).get("key"))
        self.poller = M.Poller(self.link, self.interval)
        if poll:
            self.poller.start()

    # ------------------------------------------------------------------------------------------------ the target
    def _pick_local(self, config: Path | None) -> str:
        """The config to attach to: --config, else one whose server of ours runs, else the most recently used."""
        cfgs = T.list_configs(self.root)
        pick = None
        if config is not None:
            pick = Path(config).resolve()
        else:
            running = T.running(self.root)
            pick = running[0][0] if running else (cfgs[0] if cfgs else None)
        if pick is None:
            self.say("no run config here: ./abyss.sh sets one up", 30)
            return "http://127.0.0.1:8080"
        self._attach(pick)
        return self.cfg["url"]

    def _attach(self, cfg_path: Path, start_at=None) -> None:
        self.cfg_path = Path(cfg_path)
        self.cfg = T.config_info(self.cfg_path)
        self.pid_info = T.read_pid(self.cfg_path)
        if start_at is None and self.pid_info and self.pid_info.get("log") == self.cfg.get("log"):
            start_at = self.pid_info.get("log_pos")
        self.log = M.EngineLog(self.cfg.get("log"), self.cfg.get("gpus") or [0], start_at=start_at) \
            if self.cfg.get("log") else None
        self.seen = {}

    def retarget(self, cfg_path: Path, start_at=None) -> None:
        """Attach to another config's server (after a start)."""
        self.poller.stop()
        self._attach(cfg_path, start_at)
        self.link = M.Link(self.cfg["url"], self.cfg.get("key"))
        self.poller = M.Poller(self.link, self.interval)
        if self.poll:
            self.poller.start()

    # ------------------------------------------------------------------------------------------------ each second
    def say(self, text: str, secs: float = 5.0) -> None:
        self.message, self.message_until = text, time.time() + secs

    def local_facts(self) -> dict:
        if self.demo:
            return self.demo_local
        if self.remote:
            return {"remote": True, "is_local": False, "pid_alive": None}
        info = self.pid_info
        return {"remote": False, "is_local": True, "pid_alive": bool(info) if self.cfg else None,
                "started": info.get("started") if info else None, "bench": T.bench_summary(self.root),
                "version": T.version(self.root), "config": self.cfg or {}}

    def tick(self) -> None:
        """The local reads between frames: the engine log, the PID file (2 s), the configs (10 s), the temp log."""
        self.tick_n += 1
        now = time.time()
        if self.log is not None and not self.demo:
            self.log.read()
        if self.bench_log is not None:
            self.bench_log.read()
        if self.cfg_path is not None and not self.demo and now - self._last_pid_check >= 2.0:
            self._last_pid_check = now
            self.pid_info = T.read_pid(self.cfg_path)
        if not self.remote and not self.demo and (self.configs is None or now - self._last_cfg_scan >= 10.0):
            self._last_cfg_scan = now
            self.scan_configs()
        if self.message and now > self.message_until:
            self.message = ""
        self.poller.all_requests = self.view == 3

    def scan_configs(self) -> None:
        out = []
        for p in T.list_configs(self.root):
            try:
                c = T.config_info(p)
            except OSError:
                continue
            c["running"] = T.read_pid(p) is not None
            c["serving"] = self.cfg is not None and c["path"] == self.cfg.get("path") and self.link.state == "live"
            out.append(c)
        self.configs = out
        self.sel[7] = min(self.sel[7], max(0, len(out) - 1))

    def view_data(self) -> dict:
        with self.poller.lock:
            v = M.derive(self.link, self.local_facts(), self.log, seen=self.seen)
        if self.configs and not self.demo:
            for c in self.configs:
                c["serving"] = self.cfg is not None and c["path"] == self.cfg.get("path") and v["conn"] == "live"
        self.thermal.row(v)
        return v

    def ui(self) -> dict:
        return {"view": self.view, "sel": self.sel.get(self.view, 0), "log": self.bench_log or self.log,
                "configs": self.configs, "help": self.help, "frozen": self.frozen, "message": self.message,
                "thermal": self.thermal.path.name if self.thermal.on else None, "tick": self.tick_n,
                "filter": self.filter, "scroll": self.log_scroll, "check": self.check,
                "config_file": (self.cfg or {}).get("file"),
                "key_hint": ("set (" + (f"${self.key_env}" if self.key_env else "the config's api_key") + ")")
                if self.link.key else "",
                "last_error": (self.log.last_error[1] if self.log and self.log.last_error else "")}

    def frame(self, w: int, h: int) -> list:
        """The whole screen as lines (--plain, --once)."""
        v = self.frozen_view if self.frozen and self.frozen_view else self.view_data()
        return D.screen(v, w, h, self.ui())

    # ------------------------------------------------------------------------------------------------ keys
    def go(self, view: int) -> None:
        self.view = view % len(D.VIEWS)
        self.help = False
        self.log_scroll = 0

    def move(self, d: int) -> None:
        if self.view == 5:
            self.log_scroll = max(0, self.log_scroll - d)
            return
        if self.view in self.sel:
            n = len((self.configs or []) if self.view == 7 else (self.link.metrics or {}).get("requests") or [])
            self.sel[self.view] = max(0, min(max(0, n - 1), self.sel[self.view] + d))

    def freeze(self, v: dict | None = None) -> None:
        self.frozen = not self.frozen
        self.frozen_view = (v or self.view_data()) if self.frozen else None

    def cycle_theme(self) -> str:
        self.theme = D.THEMES[(D.THEMES.index(self.theme) + 1) % len(D.THEMES)]
        return self.theme

    def toggle_thermal(self) -> None:
        p = self.thermal.toggle()
        self.say(f"temperature log: {p}" if p else "temperature log stopped", 6)

    def bench_line(self) -> str | None:
        b = T.bench_summary(self.root) if not self.demo else (self.demo_local or {}).get("bench")
        return b["line"] if b else None

    def check_api(self) -> None:
        """`h` on Connect: GET /v1/models with the key - the answer and how long it took."""
        t0 = time.time()
        req = urllib.request.Request(self.link.base + "/v1/models",
                                     headers={"Authorization": f"Bearer {self.link.key}"} if self.link.key else {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode() or "{}")
            ids = [m.get("id") for m in data.get("data") or []]
            self.check = f"ok: {len(ids)} model(s) {', '.join(str(i) for i in ids)[:40]} in {(time.time() - t0) * 1000:.0f} ms"
        except urllib.error.HTTPError as e:
            e.close()
            self.check = f"failed: /v1/models answered {e.code}" + (" (the API key)" if e.code in (401, 403) else "")
        except (OSError, ValueError) as e:
            self.check = f"failed: {getattr(e, 'reason', e)}"

    def open_browser(self) -> str:
        url = self.link.base + "/"
        if not (os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or
                sys.platform == "darwin"):
            return f"no desktop here - open {url}"
        try:
            webbrowser.open(url)
            return f"opened {url}"
        except webbrowser.Error:
            return f"open {url}"

    # ------------------------------------------------------------------------------------------------ start / stop
    def can_manage(self) -> str | None:
        """None when this Terminal may start / stop servers; else why not."""
        if self.demo:
            return "demo: nothing is started or stopped"
        if self.remote:
            return "remote: read-only (stop it on its PC, or POST /api/shutdown with its key)"
        if self.busy:
            return f"busy: {self.busy}"
        return None

    def _run(self, what: str, fn) -> None:
        self.busy = what
        self.say(what + " ...", 600)

        def go():
            try:
                msg = fn()
            except Exception as e:  # noqa: BLE001 - shown, never a crash of the screen
                msg = f"{what} failed: {e}"
            self.busy = None
            self.say(msg or "done", 10)
        threading.Thread(target=go, daemon=True).start()

    def stop_server(self, then=None) -> None:
        if (why := self.can_manage()) is not None:
            self.say(why)
            return
        cfg_path, link = self.cfg_path, self.link

        def fn():
            msg = T.stop(cfg_path, link if link.state in ("live", "stale", "not_abyss") else None,
                         say=lambda s: self.say(s, 120)) if cfg_path else "no config"
            self.started_by_us = False
            self.pid_info = None
            if then:
                return then() or msg
            return msg
        self._run("stopping the server", fn)

    def start_server(self, cfg_path: Path) -> None:
        """Start `cfg_path`'s server detached - after stopping the one this Terminal shows, if it runs."""
        if (why := self.can_manage()) is not None:
            self.say(why)
            return
        info = T.config_info(cfg_path)
        if info["missing"]:
            self.say(f"{info['file']}: missing {', '.join(info['missing'])} - {T.ME} --setup repairs it", 10)
            return

        def launch():
            p = T.start_detached(cfg_path, root=self.root)
            self.retarget(cfg_path, start_at=p.get("log_pos"))
            self.pid_info = p
            self.started_by_us = True
            self.view = 0
            return f"starting {info['file']} (detached: it keeps serving after the Terminal closes)"
        running = self.pid_info or self.link.state == "live"
        if running and self.cfg_path is not None:
            self.stop_server(then=launch)
        else:
            self._run("starting the server", launch)

    def run_bench(self, cfg_path: Path) -> None:
        """`B`: ./abyss.sh --bench of a config, detached, its output followed in the Logs view."""
        if (why := self.can_manage()) is not None:
            self.say(why)
            return

        def launch():
            out = Path(cfg_path).with_suffix(".bench.log")
            import subprocess
            kw = {"creationflags": 0x00000008 | 0x00000200} if T.WIN else {"start_new_session": True}
            cmd = [sys.executable, str(self.root / "abyss.py"), "--bench", "--config", str(cfg_path), "--plain"]
            with open(out, "wb") as f:
                T._CHILDREN.append(subprocess.Popen(cmd, cwd=str(self.root), stdin=subprocess.DEVNULL, stdout=f,
                                                    stderr=subprocess.STDOUT, **kw,
                                                    env=dict(os.environ, MAYA_SH="1", PYTHONUNBUFFERED="1")))
            self.bench_log = M.EngineLog(out, start_at=0)
            self.view = 5
            return f"--bench running: its output in Logs ({out.name}); abyss-bench.txt when it ends"
        if self.pid_info or self.link.state == "live":
            self.stop_server(then=launch)
        else:
            self._run("starting --bench", launch)

    def close(self, stop: bool = False) -> str | None:
        """The Terminal ends: the server keeps serving unless `stop`."""
        msg = None
        if stop and self.cfg_path is not None and not self.demo:
            msg = T.stop(self.cfg_path, self.link if self.link.state == "live" else None)
        self.poller.stop()
        if self.thermal.on:
            self.thermal.toggle()
        if self.fake is not None:
            try:
                self.fake.down()
            except OSError:
                pass
        return msg


# ------------------------------------------------------------------------------------------------ --plain
def _keys_posix():
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def read(timeout):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = os.read(fd, 8).decode(errors="ignore")
        return {"\x1b[C": "right", "\x1b[D": "left", "\x1b[A": "up", "\x1b[B": "down"}.get(ch, ch[:1])
    return read, lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _keys_windows():
    import msvcrt

    def read(timeout):
        end = time.time() + timeout
        while time.time() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    return {"M": "right", "K": "left", "H": "up", "P": "down"}.get(msvcrt.getwch())
                return ch
            time.sleep(0.05)
        return None
    return read, lambda: None


def run_plain(s: Session) -> int:
    """No Textual: the same screen as text, drawn again every interval (ANSI; colour unless NO_COLOR or not a TTY).
    Keys: 1-8 / ← → views, ↑↓ rows, r poll now, space freeze, t theme, L temperature log, ? help, q quit."""
    if os.name == "nt":
        os.system("")                              # (turns on the console's VT sequences)
    tty_ = sys.stdout.isatty()
    color = tty_ and not os.environ.get("NO_COLOR")
    if not sys.stdin.isatty():
        print(D.ansi(s.frame(*shutil.get_terminal_size((120, 40))), s.theme, color))
        return 0
    read, restore = (_keys_windows if os.name == "nt" else _keys_posix)()
    sys.stdout.write("\x1b[?1049h\x1b[?25l")
    try:
        while True:
            s.tick()
            w, h = shutil.get_terminal_size((100, 30))
            sys.stdout.write("\x1b[H\x1b[2J" + D.ansi(s.frame(w, h - 1), s.theme, color))
            sys.stdout.flush()
            k = read(s.interval)
            if k is None:
                continue
            if k in ("q", "\x03"):
                break
            if k.isdigit() and 1 <= int(k) <= len(D.VIEWS):
                s.go(int(k) - 1)
            elif k == "right":
                s.go(s.view + 1)
            elif k == "left":
                s.go(s.view - 1)
            elif k in ("up", "down"):
                s.move(-1 if k == "up" else 1)
            elif k == "r":
                s.poller.kick()
            elif k == " ":
                s.freeze()
            elif k == "t":
                s.cycle_theme()
            elif k == "L":
                s.toggle_thermal()
            elif k == "?":
                s.help = not s.help
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        restore()
    stop = False
    if s.started_by_us and s.pid_info:
        try:
            stop = input("Keep the server running? [Y/n] ").strip().lower().startswith("n")
        except (EOFError, KeyboardInterrupt):
            stop = False
    msg = s.close(stop)
    if msg:
        print(msg)
    return 0


def once(s: Session, as_json: bool) -> int:
    """One reading, printed: the screen's text (the Overview), or the View as JSON."""
    deadline = time.time() + 6
    while s.poller.polls == 0 and time.time() < deadline:
        time.sleep(0.05)
    s.tick()
    if as_json:
        v = s.view_data()
        v.pop("_keyed", None)
        print(json.dumps(v, indent=1, default=str))
    else:
        w, h = shutil.get_terminal_size((120, 40))
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(D.ansi(s.frame(w, h - 1), s.theme, color))
    s.close()
    return 0


def remote_warning(url: str | None, key_env: str | None) -> str | None:
    """A Bearer key over plain http to another PC can be read on the way: say so (Abyss has no TLS)."""
    if not url or not key_env or not url.startswith("http://"):
        return None
    host = url[7:].split("/")[0].rsplit(":", 1)[0]
    if host in LOCAL_HOSTS or host.startswith("127."):
        return None
    return (f"warning: {url} is plain http - the API key from ${key_env} travels unencrypted. Use Tailscale, or an "
            f"SSH tunnel: ssh -L 8082:127.0.0.1:8082 <host>, then --url http://127.0.0.1:8082")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="abyss_terminal", description="Abyss Terminal: watch and run an Abyss server.")
    ap.add_argument("--url", help="a server elsewhere, e.g. http://192.168.1.20:8082 (read-only)")
    ap.add_argument("--key-env", metavar="NAME", help="the environment variable that holds its API key (the key "
                                                       "itself is never stored or passed on the command line)")
    ap.add_argument("--config", type=Path, help="this PC's run config to attach to (default: the running one, else "
                                                "the most recently used)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between readings (0.5-10, default 1)")
    ap.add_argument("--plain", action="store_true", help="no Textual: plain text, redrawn every interval")
    ap.add_argument("--once", action="store_true", help="print one reading and exit")
    ap.add_argument("--json", action="store_true", help="print one reading as JSON and exit")
    ap.add_argument("--demo", metavar="SCENARIO", help="recorded readings, no server: r9700, lowspec, split, "
                                                       "loading, down")
    return ap


def main(argv=None, textual: bool | None = None) -> int:
    a = parser().parse_args(argv)
    warn = remote_warning(a.url, a.key_env)
    if warn:
        print(warn, file=sys.stderr)
    s = Session(a.url, a.key_env, a.config, a.interval, a.demo)
    if a.once or a.json:
        return once(s, a.json)
    if textual is None:
        textual = not a.plain and sys.stdout.isatty() and sys.stdin.isatty() and os.environ.get("TERM") != "dumb"
    if textual:
        try:
            import terminal_app
        except ImportError as e:
            print(f"Textual is not available ({e}): plain text instead (./abyss.sh --setup installs it)",
                  file=sys.stderr)
            time.sleep(1.5)
        else:
            try:
                return terminal_app.run(s)
            except Exception as e:  # noqa: BLE001 - a console Textual cannot drive (old conhost): plain text
                print(f"the screen failed ({e}): plain text instead (--plain skips the screen)", file=sys.stderr)
                time.sleep(1.5)
    return run_plain(s)


if __name__ == "__main__":
    sys.exit(main())

"""Abyss Terminal's screen (Textual, from third_party/wheels): the header, the tabs, the view and the keys, drawn from
terminal_draw's lines once a second; the questions (quit: keep serving or stop; start / stop / bench) as small boxes
over it.  abyss_terminal.Session holds everything else - this file only draws it and maps the keys.
"""
from __future__ import annotations

import os
from pathlib import Path

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

import terminal_draw as D

_STYLES: dict = {}


def rich_style(style: str, theme: str) -> Style:
    """A terminal_draw style ("accent b r") as Rich's, in `theme`'s palette (mono: the flags only)."""
    key = (style, theme)
    if key not in _STYLES:
        role, *flags = (style or "ink").split()
        fl = "".join(flags)
        pal = D.PALETTES.get(theme)
        _STYLES[key] = Style(color=pal.get(role, pal["ink"]) if pal else None, bold="b" in fl, italic="i" in fl,
                             reverse="r" in fl, dim="d" in fl, underline="u" in fl)
    return _STYLES[key]


def rich_text(lines: list, theme: str) -> Text:
    out = Text(no_wrap=True, overflow="crop", end="")
    for i, ln in enumerate(lines):
        if i:
            out.append("\n")
        for text, style in ln:
            out.append(text, rich_style(style, theme))
    return out


class Choice(ModalScreen):
    """A question over the screen: `options` to pick from (Enter), Esc = the last one (back / no)."""

    DEFAULT_CSS = """
    Choice { align: center middle; background: $background 70%; }
    #q { width: 72; max-width: 96%; height: auto; border: round #00f0ff; background: #05070d; padding: 1 2;
         border-title-color: #00f0ff; border-title-style: bold; }
    #q OptionList, #q OptionList:focus { height: auto; margin-top: 1; border: none; padding: 0;
                                         background: #05070d; background-tint: #05070d 0%; }
    #q OptionList > .option-list--option-highlighted,
    #q OptionList:focus > .option-list--option-highlighted { background: #00f0ff; color: #05070d; text-style: bold; }
    """
    BINDINGS = [Binding("escape", "back", show=False)]

    def __init__(self, title: str, text: str, options: list):
        super().__init__()
        self.q_title, self.text, self.options = title, text, options

    def compose(self) -> ComposeResult:
        with Vertical(id="q"):
            yield Static(Text(self.text, style="#9fc3cc"))
            yield OptionList(*self.options)

    def on_mount(self) -> None:
        self.query_one("#q").border_title = f" {self.q_title} "

    def action_back(self) -> None:
        self.dismiss(len(self.options) - 1)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option_index)


class Ask(ModalScreen):
    """A line of text (the Logs filter): Enter keeps it, Esc clears it."""

    DEFAULT_CSS = """
    Ask { align: center middle; background: $background 60%; }
    #a { width: 60; max-width: 96%; height: auto; border: round #00f0ff; background: #05070d; padding: 0 1;
         border-title-color: #00f0ff; }
    #a Input { border: none; background: #05070d; }
    """
    BINDINGS = [Binding("escape", "clear", show=False)]

    def __init__(self, title: str, value: str = ""):
        super().__init__()
        self.a_title, self.value = title, value

    def compose(self) -> ComposeResult:
        with Vertical(id="a"):
            yield Input(value=self.value, placeholder="text, or a kind: stat spec warn error watchdog load")

    def on_mount(self) -> None:
        self.query_one("#a").border_title = f" {self.a_title} "
        self.query_one(Input).focus()

    def action_clear(self) -> None:
        self.dismiss("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())


class TerminalApp(App):
    CSS = """
    Screen { background: #05070d; layers: base; }
    #head { height: 3; }
    #body { height: 1fr; scrollbar-size-vertical: 1; scrollbar-color: #174652; scrollbar-color-hover: #00a8c0;
            scrollbar-background: #05070d; }
    #view { height: auto; }
    #foot { height: 1; }
    """
    TITLE = "Abyss Terminal"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding(str(i + 1), f"go({i})", show=False) for i in range(8)] + [
        Binding("right", "go_by(1)", show=False), Binding("left", "go_by(-1)", show=False),
        Binding("tab", "go_by(1)", show=False, priority=True), Binding("shift+tab", "go_by(-1)", show=False,
                                                                       priority=True),
        Binding("up", "move(-1)", show=False), Binding("down", "move(1)", show=False),
        Binding("pageup", "page(-1)", show=False), Binding("pagedown", "page(1)", show=False),
        Binding("home", "home", show=False), Binding("end", "end", show=False),
        Binding("enter", "enter", show=False), Binding("s", "stop", show=False), Binding("B", "bench", show=False),
        Binding("y", "copy_bench", show=False), Binding("c", "copy_api", show=False),
        Binding("h", "check", show=False), Binding("b", "browser", show=False), Binding("r", "refresh", show=False),
        Binding("space", "freeze", show=False), Binding("t", "theme", show=False), Binding("L", "thermal", show=False),
        Binding("slash", "filter", show=False), Binding("escape", "escape", show=False),
        Binding("question_mark", "help", show=False), Binding("f1", "help", show=False),
        Binding("q", "quit_ask", show=False), Binding("ctrl+c", "quit_ask", show=False, priority=True),
    ]

    def __init__(self, session, anim: bool = True):
        super().__init__()
        self.s = session
        self.anim = anim and not os.environ.get("ABYSS_TERMINAL_STILL")
        self.shown = (None, None, None)
        self.result = "keep"
        self.v = None

    def compose(self) -> ComposeResult:
        yield Static(id="head")
        with VerticalScroll(id="body"):
            yield Static(id="view")
        yield Static(id="foot")

    def on_mount(self) -> None:
        body = self.query_one("#body")
        body.can_focus = False
        for w in self.query(Static):
            w.auto_links = False
        self.apply_theme()
        self.redraw()
        self.set_interval(0.25, self._early, repeat=8)    # (the first reading lands within a second)
        self.set_interval(self.s.interval, self.redraw)

    def _early(self) -> None:
        if self.s.poller.polls <= 1:
            self.redraw()

    def apply_theme(self) -> None:
        pal = D.PALETTES.get(self.s.theme)
        bg = pal["bg"] if pal else "#000000"
        self.screen.styles.background = bg
        self.query_one("#body").styles.scrollbar_background = bg
        self.shown = (None, None, None)

    # ------------------------------------------------------------------------------------------------ drawing
    def redraw(self, poll: bool = True) -> None:
        s = self.s
        if poll:
            s.tick()
        if s.frozen and s.frozen_view:
            v = s.frozen_view
        else:
            v = s.view_data()
        self.v = v
        w, h = self.size.width, self.size.height
        ui = s.ui()
        ui["anim"] = self.anim
        head = D.header(v, w, s.tick_n, self.anim) + [D.tabs(s.view, w)]
        body_h = max(1, h - 4)
        body = D.help_lines(w) if s.help else D.DRAW[s.view](v, w, body_h, ui)
        if len(body) > body_h:                          # (a scrollbar takes the last column)
            body = D.help_lines(w - 1) if s.help else D.DRAW[s.view](v, w - 1, body_h, ui)
            bw = w - 1
        else:
            bw = w
        body = [D.fit(x, bw) for x in body]
        foot = [D.footer(v, w, s.frozen, s.message, s.thermal.on)]
        for wid, lines, k in (("#head", head, 0), ("#view", body, 1), ("#foot", foot, 2)):
            key = (s.theme, tuple(tuple(x) for x in lines))
            if self.shown[k] != key:
                self.query_one(wid, Static).update(rich_text(lines, s.theme))
                self.shown = tuple(key if i == k else x for i, x in enumerate(self.shown))

    def on_resize(self, event) -> None:
        self.redraw(poll=False)

    def plain(self) -> str:
        """The screen's text as drawn (the tests read it)."""
        return "\n".join(str(self.query_one(w, Static).render()) for w in ("#head", "#view", "#foot"))

    # ------------------------------------------------------------------------------------------------ keys
    def action_go(self, i: int) -> None:
        self.s.go(i)
        self.query_one("#body").scroll_home(animate=False)
        self.redraw(poll=False)

    def action_go_by(self, d: int) -> None:
        self.action_go((self.s.view + d) % len(D.VIEWS))

    def action_move(self, d: int) -> None:
        if self.s.view in (3, 5, 7):
            self.s.move(d)
        else:
            body = self.query_one("#body")
            body.scroll_relative(y=d, animate=False)
        self.redraw(poll=False)

    def action_page(self, d: int) -> None:
        body = self.query_one("#body")
        (body.scroll_page_down if d > 0 else body.scroll_page_up)(animate=False)
        if self.s.view == 5:
            self.s.move(d * 10)
            self.redraw(poll=False)

    def action_home(self) -> None:
        self.query_one("#body").scroll_home(animate=False)

    def action_end(self) -> None:
        self.query_one("#body").scroll_end(animate=False)

    def action_refresh(self) -> None:
        self.s.poller.kick()
        self.s.say("polling now", 2)
        self.set_timer(0.4, lambda: self.redraw())

    def action_freeze(self) -> None:
        self.s.freeze(self.v)
        self.redraw(poll=False)

    def action_theme(self) -> None:
        self.s.say(f"theme: {self.s.cycle_theme()}", 3)
        self.apply_theme()
        self.redraw(poll=False)

    def action_thermal(self) -> None:
        self.s.toggle_thermal()
        self.redraw(poll=False)

    def action_help(self) -> None:
        self.s.help = not self.s.help
        self.redraw(poll=False)

    def action_escape(self) -> None:
        if self.s.help:
            self.s.help = False
        elif self.s.filter:
            self.s.filter = ""
        self.redraw(poll=False)

    def action_filter(self) -> None:
        if self.s.view != 5:
            return

        def done(value):
            self.s.filter = value or ""
            self.redraw(poll=False)
        self.push_screen(Ask("Filter the log", self.s.filter), done)

    def action_copy_bench(self) -> None:
        line = self.s.bench_line()
        if line:
            self.copy_to_clipboard(line)
            self.s.say("bench line copied (if the terminal allows OSC 52)", 5)
        else:
            self.s.say(f"no abyss-bench.txt yet: {D.DASH} B on Models runs --bench", 5)
        self.redraw(poll=False)

    def action_copy_api(self) -> None:
        self.copy_to_clipboard(self.s.link.base + "/v1")
        self.s.say("API address copied (if the terminal allows OSC 52)", 5)
        self.redraw(poll=False)

    def action_check(self) -> None:
        if self.s.view == 6:
            self.s.check = "checking ..."
            self.run_worker(self._check, thread=True)

    def _check(self) -> None:
        self.s.check_api()
        self.call_from_thread(self.redraw, False)

    def action_browser(self) -> None:
        self.s.say(self.s.open_browser(), 6)
        self.redraw(poll=False)

    def _selected_config(self):
        cfgs = self.s.configs or []
        i = self.s.sel.get(7, 0)
        return cfgs[i] if 0 <= i < len(cfgs) else None

    def action_enter(self) -> None:
        s = self.s
        if s.view == 7:
            c = self._selected_config()
            if c is None:
                return
            if (why := s.can_manage()) is not None:
                s.say(why)
                return self.redraw(poll=False)
            current = (s.cfg or {}).get("file")
            running = bool(s.pid_info) or s.link.state == "live"
            text = (f"Start {c['file']}?" + (f" {current} stops first (its GPU memory is freed)." if running and current
                                             else "") + " It runs detached: closing the Terminal keeps it serving.")
            self.push_screen(Choice("Start a model", text, ["Start", "Back"]),
                             lambda i: i == 0 and (s.start_server(Path(c["path"])),
                                                   self.redraw(poll=False)))
        elif s.view == 0 and self.v and self.v["conn"] == "stopped" and s.cfg_path is not None:
            self.push_screen(Choice("Start the server", f"Start {s.cfg['file']}? It runs detached.",
                                    ["Start", "Back"]),
                             lambda i: i == 0 and (s.start_server(s.cfg_path), self.redraw(poll=False)))

    def action_stop(self) -> None:
        s = self.s
        if (why := s.can_manage()) is not None:
            s.say(why)
            return self.redraw(poll=False)
        if not (s.pid_info or s.link.state in ("live", "stale")):
            s.say("no server runs for this config")
            return self.redraw(poll=False)
        self.push_screen(Choice("Stop the server", f"Stop {(s.cfg or {}).get('file')}? The request in progress is cut "
                                "and the model leaves the GPU.", ["Stop it", "Back"]),
                         lambda i: i == 0 and (s.stop_server(), self.redraw(poll=False)))

    def action_bench(self) -> None:
        s = self.s
        c = self._selected_config() if s.view == 7 else s.cfg
        if c is None:
            return
        if (why := s.can_manage()) is not None:
            s.say(why)
            return self.redraw(poll=False)
        self.push_screen(Choice("Run --bench", f"Run ./abyss.sh --bench on {c['file']}? The server stops first (the "
                                "bench loads the model itself, a few minutes).", ["Run it", "Back"]),
                         lambda i: i == 0 and (s.run_bench(Path(c["path"])),
                                               self.redraw(poll=False)))

    def action_quit_ask(self) -> None:
        s = self.s
        if s.started_by_us and (s.pid_info or s.link.state == "live"):
            def done(i):
                if i == 0:
                    self.result = "keep"
                    self.exit("keep")
                elif i == 1:
                    self.result = "stop"
                    self.exit("stop")
            self.push_screen(Choice("Quit Abyss Terminal", "This Terminal started the server. Keep it serving in the "
                                    "background, or stop it and free the GPU memory?",
                                    ["Keep serving (background)", "Stop the server", "Back"]), done)
        else:
            self.exit("keep")


def run(session) -> int:
    app = TerminalApp(session)
    result = app.run()
    stop = result == "stop"
    if stop:
        print("Stopping the server (it ends its engine and frees the GPU memory) ...")
    msg = session.close(stop)
    if msg:
        print(msg)
    elif session.started_by_us and not stop:
        print(f"The server keeps serving: {session.link.base}/ - ./abyss.sh --terminal attaches again.")
    return 0

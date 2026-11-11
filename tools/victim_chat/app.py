#!/usr/bin/env python3
"""Victim Chat — a desktop chat app for the local victim model.

NOT part of the study protocol. Measurements go through the harness; this is the
by-hand tool. It imports the same `harness.model.VictimModel`, so the weights,
quantization and image-token caps are byte-for-byte what the attacks run against
— only the conversation shape differs (multi-turn and streamed, where the
harness's single-turn choke point is deliberately neither).

Tkinter on purpose: it ships with Python, so the pinned CUDA venv gains no new
package, and there is no server, no port and no browser involved. The model's
output is attacker-controlled by construction in this project; here it is only
ever inserted as text into a Text widget, never interpreted as markup.

Conversations persist to chats/ so previous chats survive a restart.

Run:  .venv/Scripts/python.exe tools/victim_chat/app.py   (or double-click Victim Chat.cmd)
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue

import sys
import threading
import time
import tkinter as tk
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[1]          # repo root: harness/ + systems/ import from here
sys.path.insert(0, str(ROOT))
CHATS_DIR = ROOT / "chats"
IMAGES_DIR = CHATS_DIR / "images"
ICON = APP_DIR / "victim_chat.ico"
LOG_DIR = ROOT / "logs"

# --------------------------------------------------------------------------
# Palette. Light, one emerald accent, defined once — every colour in the app
# comes from here so a change is a one-line change.
# --------------------------------------------------------------------------
CANVAS      = "#FFFFFF"   # conversation ground
SIDEBAR     = "#F7F9F8"
HOVER       = "#EDF2EF"
BORDER      = "#E5E7EB"
INK         = "#111827"   # primary text   (15.3:1 on white)
INK_SOFT    = "#6B7280"   # secondary text (4.8:1 on white)
ACCENT      = "#059669"   # emerald 600 — dots, focus, highlights
ACCENT_DEEP = "#047857"   # emerald 700 — solid buttons (white text, 5.1:1)
ACCENT_SOFT = "#ECFDF5"   # emerald 50  — the user's bubble
ACCENT_LINE = "#A7F3D0"
BUBBLE_INK  = "#064E3B"
DANGER      = "#DC2626"
WARN        = "#D97706"

COLUMN_MAX = 720          # readable measure; the conversation never gets wider
GAP = 8                   # 8px spacing rhythm

SYSTEM_PRESETS: dict[str, str] = {
    "Plain assistant": "You are a helpful assistant.",
}


def _load_victim_presets() -> None:
    """Offer the study's real victim prompts, so chatting and measuring share
    wording. Import is soft: the window must still open if the harness moves."""
    try:
        from systems.system1_direct.victim.image_description import DEFENSES
    except Exception:                                     # noqa: BLE001
        return
    labels = {"off": "Victim app - defense off",
              "naive": "Victim app - naive defense",
              "spotlight": "Victim app - spotlight defense"}
    for key, prompt in DEFENSES.items():
        SYSTEM_PRESETS[labels.get(key, f"Victim app - {key}")] = prompt


_load_victim_presets()
DEFAULT_SYSTEM = SYSTEM_PRESETS["Plain assistant"]


# --------------------------------------------------------------------------
# Conversation storage
# --------------------------------------------------------------------------
@dataclass
class Turn:
    """One message. `image_file` rides with the user turn it was attached to, so
    reloading a chat reproduces the exact multimodal context."""
    role: str                        # "user" | "assistant"
    text: str = ""
    image_file: str | None = None

    def load_image(self) -> Image.Image | None:
        if not self.image_file:
            return None
        path = IMAGES_DIR / self.image_file
        if not path.is_file():
            return None
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        except Exception:                                 # noqa: BLE001
            return None


@dataclass
class Conversation:
    cid: str
    title: str = "New chat"
    system: str = DEFAULT_SYSTEM
    temperature: float = 0.7
    max_new_tokens: int = 512
    created: str = ""
    updated: str = ""
    turns: list[Turn] = None                              # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.turns is None:
            self.turns = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.created = self.created or now
        self.updated = self.updated or now

    @staticmethod
    def new() -> "Conversation":
        return Conversation(cid=uuid.uuid4().hex[:12])

    @property
    def path(self) -> Path:
        return CHATS_DIR / f"{self.cid}.json"

    def save(self) -> None:
        # An untouched chat is not worth a file; only real conversations persist.
        if not self.turns:
            return
        CHATS_DIR.mkdir(parents=True, exist_ok=True)
        self.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = asdict(self) | {"turns": [asdict(t) for t in self.turns]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def delete(self) -> None:
        for turn in self.turns:
            if turn.image_file:
                (IMAGES_DIR / turn.image_file).unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)

    def retitle_from_first_message(self) -> None:
        first = next((t.text for t in self.turns if t.role == "user" and t.text), "")
        if first:
            flat = " ".join(first.split())
            self.title = flat[:26] + ("…" if len(flat) > 26 else "")

    def store_image(self, img: Image.Image) -> str:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{self.cid}_{len(self.turns)}_{uuid.uuid4().hex[:6]}.png"
        img.save(IMAGES_DIR / name, format="PNG")
        return name


def load_conversations() -> list[Conversation]:
    """Newest first. A corrupt file is skipped, never fatal."""
    out: list[Conversation] = []
    if not CHATS_DIR.is_dir():
        return out
    for path in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            turns = [Turn(**t) for t in data.pop("turns", [])]
            data.pop("path", None)
            out.append(Conversation(turns=turns, **data))
        except Exception:                                 # noqa: BLE001
            logging.warning("skipping unreadable chat %s", path.name)
    return sorted(out, key=lambda c: c.updated, reverse=True)


# --------------------------------------------------------------------------
# Model worker — everything touching CUDA runs off the UI thread and reports
# back through a queue the Tk main loop drains. Tk is not thread-safe; no
# worker ever touches a widget.
# --------------------------------------------------------------------------
class ModelWorker:
    def __init__(self, events: "queue.Queue[tuple]") -> None:
        self.events = events
        self.model = None
        self.cancel = threading.Event()
        self.busy = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from harness.model import VictimModel
            started = time.time()
            self.model = VictimModel()
            self.events.put(("ready", self.model.describe_config(),
                             round(time.time() - started, 1)))
        except Exception as exc:                          # noqa: BLE001
            logging.exception("model load failed")
            self.events.put(("load_error", f"{type(exc).__name__}: {exc}"))

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _stopping_criteria(self):
        """Make Stop actually end generation instead of only muting the stream.

        Guarded: if transformers moves this API, chat keeps working and we lose
        early cancellation rather than crashing the app.
        """
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            cancel = self.cancel

            class _Cancelled(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    return cancel.is_set()

            return StoppingCriteriaList([_Cancelled()])
        except Exception:                                 # noqa: BLE001
            return None

    def generate(self, messages: list[dict], temperature: float,
                 max_new_tokens: int) -> None:
        """Stream one reply. Fire-and-forget: results arrive as queue events."""
        if not self.busy.acquire(blocking=False):
            self.events.put(("error", "still working on the previous message"))
            return
        threading.Thread(target=self._generate,
                         args=(messages, temperature, max_new_tokens),
                         daemon=True).start()

    def _generate(self, messages: list[dict], temperature: float,
                  max_new_tokens: int) -> None:
        try:
            import torch
            from transformers import TextIteratorStreamer
            from qwen_vl_utils import process_vision_info

            self.cancel.clear()
            vm = self.model

            text = vm.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vm.processor(text=[text], images=image_inputs,
                                  videos=video_inputs, padding=True,
                                  return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(vm.processor.tokenizer,
                                            skip_prompt=True,
                                            skip_special_tokens=True)
            gen_kwargs = dict(**inputs, streamer=streamer,
                              max_new_tokens=max_new_tokens,
                              do_sample=temperature > 0)
            if gen_kwargs["do_sample"]:
                gen_kwargs |= {"temperature": temperature, "top_p": 0.9}
            criteria = self._stopping_criteria()
            if criteria is not None:
                gen_kwargs["stopping_criteria"] = criteria

            failure: list[str] = []

            def _run() -> None:
                try:
                    with torch.inference_mode():
                        vm.model.generate(**gen_kwargs)
                except Exception as exc:                  # noqa: BLE001
                    failure.append(f"{type(exc).__name__}: {exc}")
                    streamer.end()

            started = time.time()
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()

            self.events.put(("start",))
            for chunk in streamer:
                self.events.put(("token", chunk))
            worker.join()

            if failure:
                self.events.put(("error", failure[0]))
            else:
                self.events.put(("done", round(time.time() - started, 1),
                                 self.cancel.is_set()))
        except Exception as exc:                          # noqa: BLE001
            logging.exception("generation failed")
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.busy.release()


# --------------------------------------------------------------------------
# Small widgets. Tk has no rounded corners, so the two shapes that carry the
# app's feel — the message bubble and the composer — are drawn on a Canvas.
# --------------------------------------------------------------------------
def round_rect(canvas: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r,
              x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
              x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kw)


class RoundedPanel(tk.Canvas):
    """A rounded, optionally outlined panel that grows with its content.

    `body` is the frame to put widgets in; the canvas tracks its size so the
    shape always fits what's inside instead of needing a hardcoded height.
    """

    def __init__(self, parent, *, fill: str, outline: str = "", radius: int = 18,
                 pad: int = 14, bg: str | None = None) -> None:
        super().__init__(parent, bg=bg or parent.cget("bg"), highlightthickness=0,
                         bd=0, height=1)
        self.fill, self.outline, self.radius, self.pad = fill, outline, radius, pad
        self._fitting = False
        self.body = tk.Frame(self, bg=fill)
        self._window = self.create_window(pad, pad, anchor="nw", window=self.body)
        self.body.bind("<Configure>", lambda e: self.refit())
        self.bind("<Configure>", lambda e: self._draw())
        self.after_idle(self.refit)

    def refit(self) -> None:
        """Size the canvas to its content.

        Driven by the body's REQUESTED height, never its mapped height: a canvas
        with no height doesn't map its window item, the child then never fires
        <Configure>, and a panel that waits for that event stays collapsed
        forever. Asking the child how big it wants to be breaks the deadlock.
        """
        if self._fitting or not self.winfo_exists():
            return
        self._fitting = True
        try:
            self.update_idletasks()
            wanted = self.body.winfo_reqheight() + self.pad * 2
            if wanted != self.winfo_reqheight():
                self.configure(height=wanted)
            self._draw()
        finally:
            self._fitting = False

    def hug(self) -> None:
        """Shrink the panel to its content — what makes a bubble sit tight around
        a short message instead of stretching the full column."""
        self.configure(width=self.body.winfo_reqwidth() + self.pad * 2)
        self.refit()

    def _draw(self) -> None:
        self.delete("shape")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        round_rect(self, 1, 1, w - 2, h - 2, self.radius, fill=self.fill,
                   outline=self.outline or self.fill, width=1, tags="shape")
        self.tag_lower("shape")
        self.itemconfigure(self._window, width=w - self.pad * 2)


class PillButton(tk.Canvas):
    """A flat rounded button. Used where a stock tk.Button's square grey box
    would break the look — the composer's send control and 'New chat'."""

    def __init__(self, parent, text: str, command, *, fill: str, fg: str,
                 hover: str, width: int, height: int = 36, radius: int = 999,
                 font=None, bg: str | None = None) -> None:
        super().__init__(parent, width=width, height=height, bd=0,
                         highlightthickness=0, bg=bg or parent.cget("bg"),
                         cursor="hand2")
        self.command, self.fill, self.hover = command, fill, hover
        self._enabled = True
        self._shape = round_rect(self, 1, 1, width - 2, height - 2,
                                 min(radius, height // 2), fill=fill, outline=fill)
        self._label = self.create_text(width / 2, height / 2, text=text, fill=fg,
                                       font=font)
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda e: self._tint(self.hover))
        self.bind("<Leave>", lambda e: self._tint(self.fill))

    def _tint(self, colour: str) -> None:
        if self._enabled:
            self.itemconfigure(self._shape, fill=colour, outline=colour)

    def _click(self, _event) -> None:
        if self._enabled:
            self.command()

    def set_text(self, text: str) -> None:
        self.itemconfigure(self._label, text=text)

    def set_enabled(self, enabled: bool, *, fill: str | None = None) -> None:
        self._enabled = enabled
        colour = fill or (self.fill if enabled else BORDER)
        self.itemconfigure(self._shape, fill=colour, outline=colour)
        self.configure(cursor="hand2" if enabled else "arrow")


class ScrollArea(tk.Frame):
    """Vertically scrollable region whose content frame is `inner`."""

    def __init__(self, parent, bg: str) -> None:
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Quiet.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._window, width=e.width))
        # Wheel is bound on hover only, so two scroll areas never fight.
        for widget in (self.canvas, self.inner):
            widget.bind("<Enter>", self._grab_wheel)
            widget.bind("<Leave>", self._release_wheel)

    def _grab_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _release_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def to_bottom(self) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------
class VictimChat:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.worker = ModelWorker(self.events)

        self.conversations = load_conversations()
        self.chat = Conversation.new()
        self.pending_image: Image.Image | None = None
        self._thumb: ImageTk.PhotoImage | None = None
        self._image_refs: list[ImageTk.PhotoImage] = []
        self._bubbles: list[tuple[tk.Widget, str, "RoundedPanel | None"]] = []   # for reflow on resize
        self._stream_widget: tk.Text | None = None
        self._stream_chunks: list[str] = []
        self._streaming = False
        self._resize_job: str | None = None
        self._model_ready = False

        self._fonts()
        self._window()
        self._sidebar()
        self._main()
        self._keys()

        self._render_chat_list()
        self._render_conversation()
        self._set_status("loading", "Loading model…")
        self._set_composer_enabled(False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(40, self._pump)

    # -- chrome ------------------------------------------------------------
    def _fonts(self) -> None:
        family = "Segoe UI" if "Segoe UI" in tkfont.families() else "Helvetica"
        self.f_body = tkfont.Font(family=family, size=11)
        self.f_bold = tkfont.Font(family=family, size=11, weight="bold")
        self.f_small = tkfont.Font(family=family, size=9)
        self.f_meta = tkfont.Font(family=family, size=9)
        self.f_head = tkfont.Font(family=family, size=14, weight="bold")
        self.f_big = tkfont.Font(family=family, size=20, weight="bold")

    def _window(self) -> None:
        self.root.title("Victim Chat")
        self.root.configure(bg=CANVAS)
        self.root.geometry("1080x780")
        self.root.minsize(720, 520)
        try:
            self.root.iconbitmap(default=str(ICON))
        except Exception:                                 # noqa: BLE001
            pass                       # a missing icon must never block startup

        style = ttk.Style(self.root)
        style.theme_use("clam")      # 'clam' honours colours; 'vista' ignores them
        style.configure("Quiet.Vertical.TScrollbar", background="#D9DEDB",
                        troughcolor=CANVAS, bordercolor=CANVAS,
                        arrowcolor=CANVAS, relief="flat", borderwidth=0)
        style.map("Quiet.Vertical.TScrollbar", background=[("active", "#BFC7C2")])
        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#FFFFFF",
                        foreground=INK, arrowcolor=INK_SOFT, bordercolor=BORDER,
                        lightcolor="#FFFFFF", darkcolor="#FFFFFF", padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", "#FFFFFF")],
                  foreground=[("readonly", INK)],
                  selectbackground=[("readonly", "#FFFFFF")],
                  selectforeground=[("readonly", INK)])
        style.configure("TSpinbox", fieldbackground="#FFFFFF", background="#FFFFFF",
                        foreground=INK, arrowcolor=INK_SOFT, bordercolor=BORDER)
        self.root.option_add("*TCombobox*Listbox.background", "#FFFFFF")
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_SOFT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", INK)

    # -- sidebar -----------------------------------------------------------
    def _sidebar(self) -> None:
        rail = tk.Frame(self.root, bg=SIDEBAR, width=248)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, width=1).pack(side="left", fill="y")

        head = tk.Frame(rail, bg=SIDEBAR)
        head.pack(fill="x", padx=GAP * 2, pady=(GAP * 2, GAP))
        tk.Label(head, text="Victim Chat", font=self.f_bold, bg=SIDEBAR,
                 fg=INK).pack(side="left")

        PillButton(rail, "+  New chat", self._new_chat, fill=ACCENT_DEEP,
                   fg="#FFFFFF", hover=ACCENT, width=216, height=38, radius=12,
                   font=self.f_body, bg=SIDEBAR).pack(padx=GAP * 2, pady=(GAP, GAP * 2))

        tk.Label(rail, text="Recents", font=self.f_small, bg=SIDEBAR, fg=INK_SOFT,
                 anchor="w").pack(fill="x", padx=GAP * 2 + 2, pady=(0, 4))

        self.chat_list = ScrollArea(rail, SIDEBAR)
        self.chat_list.pack(fill="both", expand=True, padx=GAP, pady=(0, GAP))

        foot = tk.Frame(rail, bg=SIDEBAR)
        foot.pack(fill="x", side="bottom", padx=GAP * 2, pady=GAP * 2)
        tk.Frame(foot, bg=BORDER, height=1).pack(fill="x", pady=(0, 12))

        status_row = tk.Frame(foot, bg=SIDEBAR)
        status_row.pack(fill="x")
        self.dot = tk.Canvas(status_row, width=8, height=8, bg=SIDEBAR,
                             highlightthickness=0)
        self.dot.pack(side="left", pady=(4, 0))
        self._dot_id = self.dot.create_oval(0, 0, 8, 8, fill=WARN, outline="")
        self.status = tk.Label(status_row, text="", font=self.f_small, bg=SIDEBAR,
                               fg=INK_SOFT, anchor="w", justify="left",
                               wraplength=180)
        self.status.pack(side="left", padx=(6, 0), fill="x", expand=True)

        for text, action in (("Manage chats", self._open_manage_chats),
                             ("Settings", self._open_settings)):
            link = tk.Label(foot, text=text, font=self.f_small, bg=SIDEBAR,
                            fg=INK_SOFT, cursor="hand2", anchor="w")
            link.pack(fill="x", pady=(GAP, 0))
            link.bind("<Button-1>", lambda e, a=action: a())
            link.bind("<Enter>", lambda e, w=link: w.configure(fg=ACCENT_DEEP))
            link.bind("<Leave>", lambda e, w=link: w.configure(fg=INK_SOFT))

    def _render_chat_list(self) -> None:
        for child in self.chat_list.inner.winfo_children():
            child.destroy()
        if not self.conversations:
            tk.Label(self.chat_list.inner, text="No previous chats yet",
                     font=self.f_small, bg=SIDEBAR, fg=INK_SOFT, anchor="w"
                     ).pack(fill="x", padx=GAP + 2, pady=6)
            return
        for convo in self.conversations:
            self._chat_row(convo)

    def _chat_row(self, convo: Conversation) -> None:
        active = convo.cid == self.chat.cid
        base = HOVER if active else SIDEBAR
        row = tk.Frame(self.chat_list.inner, bg=base, cursor="hand2")
        row.pack(fill="x", pady=1)

        label = tk.Label(row, text=convo.title, font=self.f_body, bg=base, fg=INK,
                         anchor="w", padx=10, pady=7)
        label.pack(side="left", fill="x", expand=True)
        close = tk.Label(row, text="✕", font=self.f_small, bg=base, fg=INK_SOFT,
                         padx=8, cursor="hand2")

        def tint(colour: str) -> None:
            for widget in (row, label, close):
                widget.configure(bg=colour)

        def enter(_e=None) -> None:
            tint(HOVER)
            close.pack(side="right")

        def leave(_e=None) -> None:
            tint(HOVER if convo.cid == self.chat.cid else SIDEBAR)
            if convo.cid != self.chat.cid:
                close.pack_forget()

        for widget in (row, label):
            widget.bind("<Enter>", enter)
            widget.bind("<Leave>", leave)
            widget.bind("<Button-1>", lambda e, c=convo: self._open_chat(c))
        close.bind("<Enter>", enter)
        close.bind("<Button-1>", lambda e, c=convo: self._delete_chat(c))
        if active:
            close.pack(side="right")

    # -- main column -------------------------------------------------------
    def _main(self) -> None:
        main = tk.Frame(self.root, bg=CANVAS)
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=CANVAS)
        header.pack(fill="x")
        self.title_label = tk.Label(header, text="New chat", font=self.f_head,
                                    bg=CANVAS, fg=INK, anchor="w")
        self.title_label.pack(side="left", padx=GAP * 3, pady=(GAP * 2, 12))
        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        # Composer packs to the bottom FIRST so it stays pinned there while the
        # transcript takes whatever height is left.
        self._composer(main)
        self.scroll = ScrollArea(main, CANVAS)
        self.scroll.pack(fill="both", expand=True)
        self.column = tk.Frame(self.scroll.inner, bg=CANVAS)
        self.column.pack(fill="both", expand=True)
        self.scroll.canvas.bind("<Configure>", self._on_resize, add="+")

    def _composer(self, parent) -> None:
        holder = tk.Frame(parent, bg=CANVAS)
        holder.pack(side="bottom", fill="x")

        # Held to the same centred measure as the conversation (see _reflow), so
        # the input reads as part of the column instead of a strip pinned to the
        # far edges of a wide window.
        wrap = tk.Frame(holder, bg=CANVAS)
        wrap.pack(fill="x", padx=GAP * 3, pady=(GAP, GAP))
        self.composer_wrap = wrap

        self.attach_chip = tk.Frame(wrap, bg=ACCENT_SOFT)
        self.thumb = tk.Label(self.attach_chip, bg=ACCENT_SOFT)
        self.thumb.pack(side="left", padx=(6, 8), pady=6)
        self.attach_name = tk.Label(self.attach_chip, text="", font=self.f_small,
                                    bg=ACCENT_SOFT, fg=BUBBLE_INK, justify="left")
        self.attach_name.pack(side="left")
        drop = tk.Label(self.attach_chip, text="✕", font=self.f_small,
                        bg=ACCENT_SOFT, fg=BUBBLE_INK, cursor="hand2", padx=10)
        drop.pack(side="right")
        drop.bind("<Button-1>", lambda e: self._clear_attachment())

        self.pill = RoundedPanel(wrap, fill="#FFFFFF", outline=BORDER, radius=22,
                                 pad=10, bg=CANVAS)
        self.pill.pack(fill="x")

        body = self.pill.body
        self.attach_btn = tk.Label(body, text="＋", font=self.f_head, bg="#FFFFFF",
                                   fg=INK_SOFT, cursor="hand2", padx=6)
        self.attach_btn.pack(side="left", padx=(2, 6))
        self.attach_btn.bind("<Button-1>", lambda e: self._pick_image())
        self.attach_btn.bind("<Enter>", lambda e: self.attach_btn.configure(fg=ACCENT_DEEP))
        self.attach_btn.bind("<Leave>", lambda e: self.attach_btn.configure(fg=INK_SOFT))

        self.entry = tk.Text(body, height=2, width=1, wrap="word", font=self.f_body,
                             bg="#FFFFFF", fg=INK, insertbackground=ACCENT_DEEP,
                             relief="flat", bd=0, highlightthickness=0,
                             padx=0, pady=6)
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<KeyRelease>", lambda e: self._grow_entry())

        self.send = PillButton(body, "Send", self._send, fill=ACCENT_DEEP,
                               fg="#FFFFFF", hover=ACCENT, width=74, height=34,
                               radius=17, font=self.f_body, bg="#FFFFFF")
        self.send.pack(side="right", padx=(GAP, 2))

        self.hint = tk.Label(holder, text="Enter to send · Shift+Enter for a new line"
                                          " · Ctrl+V pastes a screenshot",
                             font=self.f_small, bg=CANVAS, fg=INK_SOFT)
        self.hint.pack(pady=(0, 12))

    def _keys(self) -> None:
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)
        self.root.bind("<Control-v>", self._paste_image)
        self.root.bind("<Control-n>", lambda e: self._new_chat())
        self.root.bind("<Escape>", lambda e: self._stop() if self._streaming else None)

    # -- rendering ---------------------------------------------------------
    def _column_width(self) -> int:
        return max(320, min(COLUMN_MAX, self.scroll.canvas.winfo_width() - GAP * 8))

    def _side_pad(self) -> int:
        return max(GAP * 2, (self.scroll.canvas.winfo_width() - self._column_width()) // 2)

    def _on_resize(self, _event=None) -> None:
        """Reflow on resize, debounced — rewrapping every bubble on each pixel of
        a drag is what makes a Tk window feel sluggish."""
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._reflow)

    def _reflow(self) -> None:
        self._resize_job = None
        pad = self._side_pad()
        width = self._column_width()
        self.column.pack_configure(padx=pad)

        # Keep the composer on the same measure as the messages above it.
        outer = self.composer_wrap.master.winfo_width()
        self.composer_wrap.pack_configure(
            padx=max(GAP * 2, (outer - width) // 2))
        self.pill.refit()
        for widget, kind, panel in self._bubbles:
            if not widget.winfo_exists():
                continue
            if kind == "bubble":
                widget.configure(wraplength=int(width * 0.72))
                if panel is not None:
                    panel.after_idle(panel.hug)
            else:
                self._autosize(widget, reset=True)

    def _autosize(self, text: tk.Text, *, reset: bool = False) -> None:
        """Height a Text widget to its wrapped content, so messages flow as one
        page instead of each getting its own scrollbar.

        Not via `count -displaylines`: Tk only counts display lines it has laid
        out, so a height-1 widget reports "1 line" for a whole paragraph and the
        message renders truncated. Growing until yview shows the end is exact.
        `reset` starts from one line so a widened window can shrink a message
        back down; streaming leaves it off, which keeps growth monotonic and
        flicker-free.
        """
        if reset:
            text.configure(height=1)
        for _ in range(12):
            text.update_idletasks()
            visible = text.yview()[1]
            if visible >= 1.0:
                return
            current = int(text.cget("height"))
            # yview's fraction says how much of the content fits, so it predicts
            # the needed height directly instead of creeping up a line at a time.
            text.configure(height=max(current + 1,
                                      min(int(current / max(visible, 0.05)) + 1,
                                          current + 40)))

    def _clear_column(self) -> None:
        for child in self.column.winfo_children():
            child.destroy()
        self._bubbles.clear()
        self._image_refs.clear()

    def _render_conversation(self) -> None:
        self._clear_column()
        self.title_label.configure(text=self.chat.title)
        if not self.chat.turns:
            self._empty_state()
        else:
            for turn in self.chat.turns:
                if turn.role == "user":
                    self._add_user(turn.text, turn.load_image())
                else:
                    widget = self._add_assistant()
                    widget.configure(state="normal")
                    widget.insert("end", turn.text)
                    widget.configure(state="disabled")
                    self._autosize(widget, reset=True)
        self.root.after(60, self._reflow)
        self.root.after(80, self.scroll.to_bottom)

    def _empty_state(self) -> None:
        pane = tk.Frame(self.column, bg=CANVAS)
        pane.pack(fill="both", expand=True, pady=(56, 0))
        badge = tk.Canvas(pane, width=44, height=44, bg=CANVAS, highlightthickness=0)
        badge.pack()
        badge.create_oval(0, 0, 44, 44, fill=ACCENT_SOFT, outline=ACCENT_LINE)
        badge.create_oval(15, 15, 29, 29, fill=ACCENT, outline="")
        tk.Label(pane, text="What can I ask the victim model?", font=self.f_big,
                 bg=CANVAS, fg=INK).pack(pady=(GAP * 2, 6))
        tk.Label(pane, text="Type below, or attach an image with ＋ to test an "
                            "injection.", font=self.f_body, bg=CANVAS,
                 fg=INK_SOFT).pack()

    def _add_user(self, text: str, image: Image.Image | None) -> None:
        row = tk.Frame(self.column, bg=CANVAS)
        row.pack(fill="x", pady=(GAP * 2, 0))

        bubble = RoundedPanel(row, fill=ACCENT_SOFT, outline=ACCENT_LINE,
                              radius=18, pad=12, bg=CANVAS)
        bubble.pack(side="right", anchor="e")

        if image is not None:
            thumb = image.copy()
            thumb.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(thumb)
            self._image_refs.append(photo)
            tk.Label(bubble.body, image=photo, bg=ACCENT_SOFT).pack(
                anchor="e", pady=(0, 8 if text else 0))

        if text:
            label = tk.Label(bubble.body, text=text, font=self.f_body,
                             bg=ACCENT_SOFT, fg=BUBBLE_INK, justify="left",
                             anchor="w", wraplength=int(self._column_width() * 0.72))
            label.pack(anchor="w")
            self._bubbles.append((label, "bubble", bubble))
        bubble.after_idle(bubble.hug)

    def _add_assistant(self) -> tk.Text:
        row = tk.Frame(self.column, bg=CANVAS)
        row.pack(fill="x", pady=(GAP * 2, 0))

        name = tk.Frame(row, bg=CANVAS)
        name.pack(fill="x", anchor="w")
        dot = tk.Canvas(name, width=8, height=8, bg=CANVAS, highlightthickness=0)
        dot.pack(side="left", pady=(3, 0))
        dot.create_oval(0, 0, 8, 8, fill=ACCENT, outline="")
        tk.Label(name, text="Qwen2.5-VL", font=self.f_small, bg=CANVAS,
                 fg=INK_SOFT).pack(side="left", padx=(6, 0))

        body = tk.Text(row, wrap="word", font=self.f_body, bg=CANVAS, fg=INK,
                       relief="flat", bd=0, highlightthickness=0, height=1, width=1,
                       padx=0, pady=4, spacing3=4, cursor="arrow",
                       selectbackground=ACCENT_SOFT, selectforeground=INK)
        body.pack(fill="x", pady=(2, 0))
        body.configure(state="disabled")
        self._bubbles.append((body, "text", None))
        return body

    def _add_note(self, text: str, colour: str) -> None:
        label = tk.Label(self.column, text=text, font=self.f_small, bg=CANVAS,
                         fg=colour, anchor="w", justify="left",
                         wraplength=self._column_width())
        label.pack(fill="x", pady=(6, 0))

    # -- conversations -----------------------------------------------------
    def _new_chat(self) -> None:
        if self._streaming:
            return
        self.chat.save()
        self._refresh_recents()
        self.chat = Conversation.new()
        self._clear_attachment()
        self._render_conversation()
        self._render_chat_list()
        self.entry.focus_set()

    def _open_chat(self, convo: Conversation) -> None:
        if self._streaming or convo.cid == self.chat.cid:
            return
        self.chat.save()
        self._refresh_recents()
        self.chat = convo
        self._clear_attachment()
        self._render_conversation()
        self._render_chat_list()

    def _delete_chat(self, convo: Conversation) -> None:
        if self._streaming:
            return
        convo.delete()
        self.conversations = [c for c in self.conversations if c.cid != convo.cid]
        if convo.cid == self.chat.cid:
            self.chat = Conversation.new()
            self._render_conversation()
        self._render_chat_list()

    def _refresh_recents(self) -> None:
        """Keep the in-memory list in step with what's on disk after a save."""
        if not self.chat.turns:
            return
        self.conversations = [c for c in self.conversations if c.cid != self.chat.cid]
        self.conversations.insert(0, self.chat)

    # -- attachments -------------------------------------------------------
    def _set_attachment(self, img: Image.Image, label: str) -> None:
        self.pending_image = img
        thumb = img.copy()
        thumb.thumbnail((40, 40))
        self._thumb = ImageTk.PhotoImage(thumb)
        self.thumb.configure(image=self._thumb)
        self.attach_name.configure(text=f"{label}   {img.width}×{img.height}")
        self.attach_chip.pack(fill="x", pady=(0, 6), before=self.pill)

    def _clear_attachment(self) -> None:
        self.pending_image = None
        self._thumb = None
        self.attach_chip.pack_forget()

    def _pick_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Attach an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                       ("All files", "*.*")])
        if path:
            self._open_image(Path(path))

    def _open_image(self, path: Path) -> None:
        try:
            img = Image.open(path)
            img.load()
            self._set_attachment(img.convert("RGB"), path.name)
        except Exception as exc:                          # noqa: BLE001
            self._add_note(f"Could not open {path.name}: {exc}", DANGER)

    def _paste_image(self, _event=None) -> None:
        try:
            data = ImageGrab.grabclipboard()
        except Exception:                                 # noqa: BLE001
            return
        if isinstance(data, Image.Image):
            self._set_attachment(data.convert("RGB"), "pasted image")
        elif isinstance(data, list) and data:
            self._open_image(Path(data[0]))

    # -- composing ---------------------------------------------------------
    def _grow_entry(self) -> None:
        """Let the input grow with the text, up to a point, like a real composer."""
        lines = int(self.entry.index("end-1c").split(".")[0])
        self.entry.configure(height=max(2, min(lines, 8)))
        self.pill.refit()

    def _on_return(self, _event):
        self._send()
        return "break"                # stop Enter also inserting a newline

    def _set_composer_enabled(self, enabled: bool) -> None:
        self.entry.configure(state="normal" if enabled else "disabled")
        self.send.set_enabled(enabled, fill=ACCENT_DEEP if enabled else BORDER)
        self.attach_btn.configure(fg=INK_SOFT if enabled else BORDER)

    def _messages(self) -> list[dict]:
        """The conversation in the model's format, images resolved from disk."""
        messages: list[dict] = [{"role": "system", "content": self.chat.system}]
        for turn in self.chat.turns:
            if turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.text})
                continue
            content: list[dict] = []
            img = turn.load_image()
            if img is not None:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": turn.text})
            messages.append({"role": "user", "content": content})
        return messages

    def _send(self) -> None:
        if self._streaming or not self._model_ready:
            return
        text = self.entry.get("1.0", "end").strip()
        image = self.pending_image
        if not text and image is None:
            return
        if not text:
            text = "Describe this image."

        first = not self.chat.turns
        if first:
            self._clear_column()

        image_file = self.chat.store_image(image) if image is not None else None
        self.chat.turns.append(Turn("user", text, image_file))
        if first:
            self.chat.retitle_from_first_message()
            self.title_label.configure(text=self.chat.title)

        self._add_user(text, image)
        self.entry.delete("1.0", "end")
        self._grow_entry()
        self._clear_attachment()

        self._streaming = True
        self._stream_chunks = []
        self._stream_widget = None
        self._set_composer_enabled(False)
        self.send.set_text("Stop")
        self.send.set_enabled(True, fill=DANGER)
        self.send.command = self._stop
        self._set_status("busy", "Thinking…")
        self.scroll.to_bottom()
        self.worker.generate(self._messages(), self.chat.temperature,
                             self.chat.max_new_tokens)

    def _stop(self) -> None:
        self.worker.cancel.set()

    def _end_stream(self) -> None:
        self._streaming = False
        self.send.set_text("Send")
        self.send.command = self._send
        self._set_composer_enabled(True)
        self.entry.focus_set()

    # -- settings ----------------------------------------------------------
    def _open_settings(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=CANVAS)
        win.transient(self.root)
        win.resizable(False, False)
        win.geometry("560x520")

        pad = {"padx": GAP * 3}
        tk.Label(win, text="Settings", font=self.f_head, bg=CANVAS, fg=INK,
                 anchor="w").pack(fill="x", pady=(GAP * 2, GAP), **pad)
        tk.Label(win, text="Applies to this chat.", font=self.f_small, bg=CANVAS,
                 fg=INK_SOFT, anchor="w").pack(fill="x", **pad)

        tk.Label(win, text="System prompt", font=self.f_bold, bg=CANVAS, fg=INK,
                 anchor="w").pack(fill="x", pady=(GAP * 2, 4), **pad)
        preset = ttk.Combobox(win, values=list(SYSTEM_PRESETS), state="readonly",
                              font=self.f_body)
        preset.pack(fill="x", **pad)
        box = tk.Text(win, height=9, wrap="word", font=self.f_body, bg="#FFFFFF",
                      fg=INK, relief="solid", bd=1, padx=10, pady=8,
                      highlightthickness=0, insertbackground=ACCENT_DEEP)
        box.pack(fill="x", pady=(6, 0), **pad)
        box.insert("1.0", self.chat.system)
        preset.bind("<<ComboboxSelected>>", lambda e: (
            box.delete("1.0", "end"), box.insert("1.0", SYSTEM_PRESETS[preset.get()])))
        tk.Label(win, text="The three “Victim app” presets are the study's real "
                           "defense prompts, read live from image_description.py.",
                 font=self.f_small, bg=CANVAS, fg=INK_SOFT, anchor="w",
                 wraplength=480, justify="left").pack(fill="x", pady=(6, 0), **pad)

        row = tk.Frame(win, bg=CANVAS)
        row.pack(fill="x", pady=(GAP * 2, 0), **pad)
        tk.Label(row, text="Temperature", font=self.f_bold, bg=CANVAS,
                 fg=INK).pack(side="left")
        temp = tk.DoubleVar(value=self.chat.temperature)
        readout = tk.Label(row, text=f"{self.chat.temperature:.2f}", font=self.f_body,
                           bg=CANVAS, fg=INK_SOFT, width=5)
        readout.pack(side="right")
        tk.Scale(row, from_=0.0, to=1.5, resolution=0.05, orient="horizontal",
                 variable=temp, showvalue=False, length=240, bg=CANVAS, fg=INK,
                 troughcolor="#E8ECEA", activebackground=ACCENT, bd=0,
                 highlightthickness=0, sliderrelief="flat",
                 command=lambda v: readout.configure(text=f"{float(v):.2f}")
                 ).pack(side="right", padx=GAP)

        row2 = tk.Frame(win, bg=CANVAS)
        row2.pack(fill="x", pady=(GAP, 0), **pad)
        tk.Label(row2, text="Max new tokens", font=self.f_bold, bg=CANVAS,
                 fg=INK).pack(side="left")
        maxtok = tk.IntVar(value=self.chat.max_new_tokens)
        ttk.Spinbox(row2, from_=32, to=2048, increment=32, width=7,
                    textvariable=maxtok, font=self.f_body).pack(side="right")
        tk.Label(win, text="0.00 is greedy decoding — the harness's existence mode.",
                 font=self.f_small, bg=CANVAS, fg=INK_SOFT, anchor="w"
                 ).pack(fill="x", pady=(6, 0), **pad)

        def save() -> None:
            self.chat.system = box.get("1.0", "end").strip() or DEFAULT_SYSTEM
            self.chat.temperature = round(float(temp.get()), 2)
            self.chat.max_new_tokens = max(32, min(int(maxtok.get()), 2048))
            self.chat.save()
            win.destroy()

        buttons = tk.Frame(win, bg=CANVAS)
        buttons.pack(fill="x", side="bottom", pady=GAP * 2, **pad)
        PillButton(buttons, "Save", save, fill=ACCENT_DEEP, fg="#FFFFFF",
                   hover=ACCENT, width=92, height=36, radius=12, font=self.f_body,
                   bg=CANVAS).pack(side="right")
        PillButton(buttons, "Cancel", win.destroy, fill="#EFF2F0", fg=INK,
                   hover=BORDER, width=92, height=36, radius=12, font=self.f_body,
                   bg=CANVAS).pack(side="right", padx=GAP)

    # -- managing saved chats ----------------------------------------------
    @staticmethod
    def _when(iso: str) -> str:
        """'2026-08-28T20:02:03+00:00' -> '28 Aug, 20:02' in local time."""
        try:
            stamp = datetime.fromisoformat(iso)
            if stamp.tzinfo is not None:
                stamp = stamp.astimezone()
            return stamp.strftime("%d %b, %H:%M")
        except ValueError:
            return ""

    def _open_manage_chats(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Manage chats")
        win.configure(bg=CANVAS)
        win.transient(self.root)
        win.geometry("620x560")
        win.minsize(480, 360)
        try:
            win.iconbitmap(str(ICON))
        except Exception:                                 # noqa: BLE001
            pass

        tk.Label(win, text="Manage chats", font=self.f_head, bg=CANVAS, fg=INK,
                 anchor="w").pack(fill="x", padx=GAP * 3, pady=(GAP * 2, 2))
        subtitle = tk.Label(win, text="", font=self.f_small, bg=CANVAS,
                            fg=INK_SOFT, anchor="w")
        subtitle.pack(fill="x", padx=GAP * 3, pady=(0, GAP))

        listing = ScrollArea(win, CANVAS)
        listing.pack(fill="both", expand=True, padx=GAP * 2, pady=(0, GAP))

        footer = tk.Frame(win, bg=CANVAS)
        footer.pack(fill="x", side="bottom", padx=GAP * 3, pady=GAP * 2)

        def render() -> None:
            for child in listing.inner.winfo_children():
                child.destroy()
            count = len(self.conversations)
            subtitle.configure(
                text=f"{count} saved chat{'' if count == 1 else 's'} in "
                     f"{CHATS_DIR.name}{chr(92)}  ·  deleting also removes any "
                     f"images that chat used")
            if not count:
                tk.Label(listing.inner, text="No saved chats.", font=self.f_body,
                         bg=CANVAS, fg=INK_SOFT, anchor="w"
                         ).pack(fill="x", padx=GAP, pady=GAP)
                wipe.set_enabled(False)
                return
            wipe.set_enabled(True, fill=DANGER)
            for convo in list(self.conversations):
                row = tk.Frame(listing.inner, bg=CANVAS)
                row.pack(fill="x", pady=2)
                text = tk.Frame(row, bg=CANVAS)
                text.pack(side="left", fill="x", expand=True)
                tk.Label(text, text=convo.title, font=self.f_body, bg=CANVAS,
                         fg=INK, anchor="w").pack(fill="x")
                images = sum(1 for t in convo.turns if t.image_file)
                detail = (f"{len(convo.turns)} messages · {self._when(convo.updated)}"
                          + (f" · {images} image{'' if images == 1 else 's'}"
                             if images else ""))
                tk.Label(text, text=detail, font=self.f_small, bg=CANVAS,
                         fg=INK_SOFT, anchor="w").pack(fill="x")

                def remove(c=convo) -> None:
                    self._delete_chat(c)
                    render()

                PillButton(row, "Delete", remove, fill="#FDECEC", fg=DANGER,
                           hover="#FBD5D5", width=82, height=32, radius=10,
                           font=self.f_small, bg=CANVAS).pack(side="right",
                                                              padx=(GAP, GAP))
                tk.Frame(listing.inner, bg=BORDER, height=1).pack(fill="x", pady=2)

        def wipe_all() -> None:
            if self._streaming:
                return
            count = len(self.conversations)
            if not count or not messagebox.askyesno(
                    "Delete all chats",
                    f"Permanently delete all {count} saved chats and their "
                    f"images?\n\nThis cannot be undone.",
                    icon="warning", parent=win):
                return
            for convo in list(self.conversations):
                self._delete_chat(convo)
            render()

        wipe = PillButton(footer, "Delete all chats", wipe_all, fill=DANGER,
                          fg="#FFFFFF", hover="#B91C1C", width=150, height=36,
                          radius=12, font=self.f_body, bg=CANVAS)
        wipe.pack(side="left")
        PillButton(footer, "Close", win.destroy, fill="#EFF2F0", fg=INK,
                   hover=BORDER, width=92, height=36, radius=12, font=self.f_body,
                   bg=CANVAS).pack(side="right")
        render()

    # -- status ------------------------------------------------------------
    def _set_status(self, kind: str, text: str) -> None:
        colour = {"loading": WARN, "ready": ACCENT, "busy": ACCENT,
                  "error": DANGER}.get(kind, INK_SOFT)
        self.dot.itemconfigure(self._dot_id, fill=colour)
        self.status.configure(text=text)

    # -- event pump --------------------------------------------------------
    def _pump(self) -> None:
        """Single place where worker output reaches widgets, on the Tk thread."""
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "ready":
                    cfg, secs = event[1], event[2]
                    self._model_ready = True
                    self._set_status("ready", f"{cfg['model_id'].split('/')[-1]} · "
                                              f"4-bit · ready in {secs}s")
                    self._set_composer_enabled(True)
                    self.entry.focus_set()

                elif kind == "load_error":
                    self._set_status("error", "Model failed to load")
                    self._add_note(f"Model load failed: {event[1]}", DANGER)

                elif kind == "start":
                    self._stream_widget = self._add_assistant()
                    self.scroll.to_bottom()

                elif kind == "token":
                    self._stream_chunks.append(event[1])
                    widget = self._stream_widget
                    if widget is not None:
                        widget.configure(state="normal")
                        widget.insert("end", event[1])
                        widget.configure(state="disabled")
                        self._autosize(widget)   # grow-only: no per-token flicker
                        self.scroll.to_bottom()

                elif kind == "done":
                    secs, cancelled = event[1], event[2]
                    reply = "".join(self._stream_chunks).strip()
                    self.chat.turns.append(Turn("assistant", reply))
                    self.chat.save()
                    self._refresh_recents()
                    self._render_chat_list()
                    self._set_status("ready", ("stopped early · " if cancelled else "")
                                     + f"replied in {secs}s")
                    self._end_stream()

                elif kind == "error":
                    self._add_note(event[1], DANGER)
                    self._set_status("error", "Generation failed")
                    if self.chat.turns and self.chat.turns[-1].role == "user":
                        self.chat.turns.pop()   # don't poison the next turn
                    self._end_stream()
        except queue.Empty:
            pass
        self.root.after(40, self._pump)

    def _on_close(self) -> None:
        self.worker.cancel.set()      # let an in-flight generation stop
        try:
            self.chat.save()
        finally:
            self.root.destroy()


def _attach_output() -> None:
    """Give the process somewhere to write when launched from the shortcut.

    pythonw.exe sets sys.stdout and sys.stderr to None. Anything that prints —
    logging, and the HF/tqdm progress bars during model load — then raises on a
    None stream, so point both at a log file instead of leaving them unset.
    """
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if sys.stdout is not None and sys.stderr is not None:
        logging.basicConfig(level=logging.INFO, format=fmt)
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Overwrite per launch: the loader's progress bars would otherwise grow this
    # without bound, and it is the current session you want when something fails.
    stream = open(LOG_DIR / "app.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    logging.basicConfig(level=logging.INFO, format=fmt, stream=stream)


def main() -> int:
    _attach_output()
    try:      # crisp text on a scaled Windows display instead of a blurry upscale
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:                                     # noqa: BLE001
        pass
    try:      # give the window its own taskbar identity, and thus its own icon
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "victimlab.chat")
    except Exception:                                     # noqa: BLE001
        pass
    root = tk.Tk()
    VictimChat(root)
    root.mainloop()
    # Hard exit: a CUDA worker mid-generation can keep the interpreter alive
    # after the window is gone, which with no console would leave a process the
    # user cannot see or close. Closing the window ends the app, full stop.
    try:
        sys.stdout.flush()
    except Exception:                                     # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

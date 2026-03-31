"""
gui.py — DroCap Desktop Interface
======================================
Run with:
  poetry run python gui.py

Requires: opencv-python, Pillow  (pip install opencv-python Pillow)
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

try:
    import cv2
    import pandas as pd
    from PIL import Image, ImageTk
    _RESULTS_AVAILABLE = True
except ImportError as _e:
    _RESULTS_AVAILABLE = False
    _RESULTS_MISSING = str(_e)

# ── Colour palette ──────────────────────────────────────────────────────────
BG        = "#0e1117"
PANEL     = "#161b24"
BORDER    = "#242c3a"
ACCENT    = "#3b8beb"
ACCENT2   = "#1d6fd8"
SUCCESS   = "#2ea66a"
WARNING   = "#e0a832"
ERROR_CLR = "#e05454"
FG        = "#d4dbe8"
FG_DIM    = "#6b7a96"
FG_LABEL  = "#8fa0bc"

FF       = "Consolas"
FONT_BODY    = (FF, 10)
FONT_LABEL   = (FF, 9)
FONT_HEADING = (FF, 11, "bold")
FONT_MONO    = (FF, 9)


# ── Shared helpers ───────────────────────────────────────────────────────────

def _browse_file(var, title, filetypes):
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if p:
        var.set(p)

def _browse_dir(var, title):
    p = filedialog.askdirectory(title=title)
    if p:
        var.set(p)

def _frame(parent, **kw):
    kw.setdefault("bg", PANEL)
    kw.setdefault("highlightbackground", BORDER)
    kw.setdefault("highlightthickness", 1)
    return tk.Frame(parent, **kw)

def _entry(parent, textvariable, width=38, **kw):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg="#1c2333", fg=FG, insertbackground=ACCENT, relief="flat",
                    font=FONT_BODY, highlightbackground=BORDER,
                    highlightthickness=1, highlightcolor=ACCENT, **kw)

def _button(parent, text, command, accent=False, small=False, **kw):
    return tk.Button(parent, text=text, command=command,
                     bg=ACCENT if accent else "#1e2a3a",
                     fg="#ffffff" if accent else FG,
                     activebackground=ACCENT2 if accent else "#263347",
                     activeforeground="#ffffff", relief="flat",
                     font=FONT_LABEL if small else FONT_BODY,
                     padx=6 if small else 10, pady=2 if small else 4,
                     cursor="hand2", **kw)

def _section_sep(parent, label):
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill="x", padx=16, pady=(14, 4))
    tk.Label(row, text=label, bg=PANEL, fg=FG_LABEL,
             font=(FF, 9, "bold")).pack(side="left")
    tk.Frame(row, bg=BORDER, height=1).pack(
        side="left", fill="x", expand=True, padx=(8, 0), pady=6)

def _file_row(parent, label, var, browse_fn, hint=""):
    outer = tk.Frame(parent, bg=PANEL)
    outer.pack(fill="x", padx=16, pady=3)
    tk.Label(outer, text=label, bg=PANEL, fg=FG_LABEL,
             font=FONT_LABEL, width=18, anchor="w").pack(side="left")
    _entry(outer, var, width=36).pack(side="left", padx=(0, 6))
    _button(outer, "Browse", browse_fn, small=True).pack(side="left")
    if hint:
        tk.Label(outer, text=hint, bg=PANEL, fg=FG_DIM,
                 font=(FF, 8)).pack(side="left", padx=(8, 0))

def _option_row(parent, label, widget_fn):
    outer = tk.Frame(parent, bg=PANEL)
    outer.pack(fill="x", padx=16, pady=3)
    tk.Label(outer, text=label, bg=PANEL, fg=FG_LABEL,
             font=FONT_LABEL, width=18, anchor="w").pack(side="left")
    widget_fn(outer)


# ── Log pane ─────────────────────────────────────────────────────────────────

class LogPane(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        hdr = tk.Frame(self, bg="#111827", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  OUTPUT LOG", bg="#111827", fg=FG_DIM,
                 font=(FF, 8, "bold")).pack(side="left", pady=6)
        _button(hdr, "Clear", self.clear, small=True).pack(side="right", padx=8, pady=4)

        self.text = tk.Text(self, bg="#0a0e16", fg=FG, font=FONT_MONO, relief="flat",
                            wrap="word", state="disabled", padx=10, pady=8,
                            cursor="arrow", selectbackground=ACCENT)
        sb = tk.Scrollbar(self, orient="vertical", command=self.text.yview,
                          bg=BG, troughcolor=BG, width=10)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

        for tag, color in (("info", FG), ("success", SUCCESS), ("warning", WARNING),
                           ("error", ERROR_CLR), ("dim", FG_DIM), ("accent", ACCENT)):
            self.text.tag_config(tag, foreground=color)

    def write(self, text, tag="info"):
        self.text.configure(state="normal")
        self.text.insert("end", text, tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def writeln(self, text, tag="info"):
        self.write(text + "\n", tag)

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


# ── Video player widget ───────────────────────────────────────────────────────

class VideoPlayer(tk.Frame):
    """
    Embedded video player using OpenCV + PIL.
    Plays the diagnostic.mp4 produced by the analyze pipeline.
    """
    _PLAY  = "▶  Play"
    _PAUSE = "⏸  Pause"

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._cap        = None
        self._playing    = False
        self._frame_idx  = 0
        self._total      = 0
        self._fps        = 30.0
        self._after_id   = None
        self._photo      = None   # prevent GC
        self._seeking    = False

        # Canvas
        self._canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        # Placeholder text
        self._canvas.create_text(
            10, 10, anchor="nw",
            text="No video loaded.\nRun Analyze to populate this view.",
            fill=FG_DIM, font=(FF, 10), tags="placeholder",
        )

        # Controls bar
        ctrl = tk.Frame(self, bg="#0a0e16", height=36)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        self._play_btn = _button(ctrl, self._PLAY, self._toggle_play, small=True)
        self._play_btn.pack(side="left", padx=(8, 4), pady=4)

        self._time_lbl = tk.Label(ctrl, text="0:00 / 0:00", bg="#0a0e16",
                                  fg=FG_DIM, font=(FF, 8))
        self._time_lbl.pack(side="right", padx=8)

        self._scrubber = tk.Scale(
            ctrl, from_=0, to=100, orient="horizontal",
            bg="#0a0e16", fg=FG_DIM, troughcolor=BORDER,
            highlightthickness=0, sliderrelief="flat",
            bd=0, showvalue=False,
            # No command= here — we use mouse bindings so programmatic
            # set() calls during playback don't trigger a seek+stop.
        )
        self._scrubber.pack(side="left", fill="x", expand=True, padx=4)
        # Only seek when the user actually drags or clicks the scrubber
        self._scrubber.bind("<ButtonPress-1>",   self._scrub_start)
        self._scrubber.bind("<ButtonRelease-1>", self._scrub_end)

    # ── Public API ──

    def load(self, path: Path) -> bool:
        self._stop()
        if self._cap:
            self._cap.release()
            self._cap = None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False

        self._cap       = cap
        self._fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_idx = 0

        self._scrubber.configure(to=max(1, self._total - 1))
        self._canvas.delete("placeholder")
        self._seek_to(0)
        return True

    def unload(self):
        self._stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._canvas.delete("all")
        self._canvas.create_text(
            10, 10, anchor="nw",
            text="No video loaded.\nRun Analyze to populate this view.",
            fill=FG_DIM, font=(FF, 10), tags="placeholder",
        )
        self._time_lbl.configure(text="0:00 / 0:00")

    # ── Internal ──

    def _toggle_play(self):
        if self._cap is None:
            return
        if self._playing:
            self._stop()
        else:
            self._playing = True
            self._play_btn.configure(text=self._PAUSE)
            self._schedule()

    def _stop(self):
        self._playing = False
        self._play_btn.configure(text=self._PLAY)
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _schedule(self):
        if not self._playing:
            return
        delay = max(1, int(1000 / self._fps))
        self._after_id = self.after(delay, self._next_frame)

    def _next_frame(self):
        if self._cap is None or not self._playing:
            return
        ok, frame = self._cap.read()
        if not ok:
            # End of video — loop back
            self._seek_to(0)
            self._stop()
            return
        self._frame_idx += 1
        self._display(frame)
        self._update_scrubber()
        self._schedule()

    def _seek_to(self, idx: int):
        if self._cap is None:
            return
        idx = max(0, min(idx, self._total - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if ok:
            self._frame_idx = idx
            self._display(frame)
            self._update_scrubber()

    def _display(self, frame_bgr):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = 640, 360

        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self._canvas.delete("frame")
        x, y = (cw - nw) // 2, (ch - nh) // 2
        self._canvas.create_image(x, y, anchor="nw",
                                  image=self._photo, tags="frame")

    def _update_scrubber(self):
        if not self._seeking:
            self._scrubber.set(self._frame_idx)
        secs  = self._frame_idx / self._fps
        total = self._total     / self._fps
        self._time_lbl.configure(
            text=f"{int(secs//60)}:{int(secs%60):02d} / "
                 f"{int(total//60)}:{int(total%60):02d}"
        )

    def _scrub_start(self, _event):
        """User pressed the scrubber — pause while dragging."""
        self._seeking = True
        self._stop()

    def _scrub_end(self, _event):
        """User released the scrubber — seek to the chosen position."""
        if self._cap is None:
            self._seeking = False
            return
        self._seek_to(int(self._scrubber.get()))
        self._seeking = False


# ── Angle table widget ────────────────────────────────────────────────────────

class AngleTable(tk.Frame):
    """
    Scrollable treeview showing angle data from angles_sagittal.csv.
    Columns: Frame | Time (s) | Hip (°) | Knee (°) | Ankle (°)
    """
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._build()

    def _build(self):
        # Style the ttk treeview to match dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Angles.Treeview",
                        background="#0f1520", foreground=FG,
                        fieldbackground="#0f1520", rowheight=22,
                        font=(FF, 9))
        style.configure("Angles.Treeview.Heading",
                        background=PANEL, foreground=FG_LABEL,
                        font=(FF, 9, "bold"), relief="flat")
        style.map("Angles.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])

        cols = ("frame", "time_s", "hip", "knee", "ankle")
        self._tree = ttk.Treeview(self, columns=cols, show="headings",
                                  style="Angles.Treeview", selectmode="browse")

        heads = {
            "frame":  ("Frame",    60),
            "time_s": ("Time (s)", 80),
            "hip":    ("Hip (°)", 100),
            "knee":   ("Knee (°)", 100),
            "ankle":  ("Ankle (°)", 100),
        }
        for col, (label, w) in heads.items():
            self._tree.heading(col, text=label)
            self._tree.column(col, width=w, anchor="center", minwidth=50)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Alternating row colours
        self._tree.tag_configure("odd",  background="#0f1520")
        self._tree.tag_configure("even", background="#131a28")

    def load(self, csv_path: Path) -> bool:
        self._tree.delete(*self._tree.get_children())
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return False

        # Find angle columns regardless of side prefix
        def _col(df, *candidates):
            for c in candidates:
                matches = [x for x in df.columns if c in x.lower()]
                if matches:
                    return matches[0]
            return None

        hip_col   = _col(df, "hip")
        knee_col  = _col(df, "knee")
        ankle_col = _col(df, "ankle")
        time_col  = _col(df, "time_s", "time")
        frame_col = _col(df, "frame")

        for i, row in df.iterrows():
            tag = "even" if i % 2 == 0 else "odd"
            frame = int(row[frame_col]) if frame_col else i
            time  = f"{row[time_col]:.3f}" if time_col else "—"
            hip   = f"{row[hip_col]:.2f}"   if hip_col   else "—"
            knee  = f"{row[knee_col]:.2f}"  if knee_col  else "—"
            ankle = f"{row[ankle_col]:.2f}" if ankle_col else "—"
            self._tree.insert("", "end", values=(frame, time, hip, knee, ankle),
                              tags=(tag,))
        return True

    def clear(self):
        self._tree.delete(*self._tree.get_children())


# ── Results tab ───────────────────────────────────────────────────────────────

class ResultsTab(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._run_dir: Path | None = None
        self._build()

    def _build(self):
        if not _RESULTS_AVAILABLE:
            tk.Label(self, text=f"Results unavailable — missing dependency:\n{_RESULTS_MISSING}\n\n"
                                 "Run:  pip install opencv-python Pillow pandas",
                     bg=BG, fg=ERROR_CLR, font=FONT_BODY, justify="left").pack(
                padx=30, pady=30, anchor="nw")
            return

        # ── Top bar: manual load ──
        top = tk.Frame(self, bg=PANEL, highlightbackground=BORDER,
                       highlightthickness=1)
        top.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(top, text="RESULTS", bg=PANEL, fg=FG,
                 font=FONT_HEADING).pack(side="left", padx=16, pady=10)

        self._dir_var = tk.StringVar()
        _entry(top, self._dir_var, width=42).pack(side="left", padx=(0, 6), pady=8)
        _button(top, "Load run dir", self._manual_load, small=True).pack(
            side="left", pady=8)
        _button(top, "Browse", lambda: (
            _browse_dir(self._dir_var, "Select run directory"),
            self._manual_load(),
        ), small=True).pack(side="left", padx=4, pady=8)

        self._info_lbl = tk.Label(top, text="", bg=PANEL, fg=FG_DIM,
                                  font=(FF, 8))
        self._info_lbl.pack(side="left", padx=10)

        # ── Sub-tab bar ──
        sub_bar = tk.Frame(self, bg="#111827")
        sub_bar.pack(fill="x", padx=20, pady=(8, 0))

        self._video_frame  = tk.Frame(self, bg=BG)
        self._angles_frame = tk.Frame(self, bg=BG)
        self._active_sub   = None

        self._sub_btns = []
        for i, (label, frame) in enumerate((
            ("  VIDEO  ", self._video_frame),
            ("  ANGLES  ", self._angles_frame),
        )):
            btn = tk.Button(sub_bar, text=label,
                            command=lambda i=i: self._select_sub(i),
                            bg="#1a2235" if i == 0 else "#111827",
                            fg=FG if i == 0 else FG_DIM,
                            relief="flat", font=(FF, 9, "bold"),
                            padx=14, pady=6, bd=0, cursor="hand2")
            btn.pack(side="left")
            self._sub_btns.append((btn, frame))

        # Show first sub-tab
        self._active_sub = 0
        self._video_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # ── Video player ──
        self._player = VideoPlayer(self._video_frame)
        self._player.pack(fill="both", expand=True, pady=(8, 0))

        # ── Angle table ──
        self._table = AngleTable(self._angles_frame)
        self._table.pack(fill="both", expand=True, padx=0, pady=(8, 0))

    def _select_sub(self, idx):
        if self._active_sub is not None:
            _, frame = self._sub_btns[self._active_sub]
            frame.pack_forget()
            btn, _ = self._sub_btns[self._active_sub]
            btn.configure(bg="#111827", fg=FG_DIM)

        self._active_sub = idx
        btn, frame = self._sub_btns[idx]
        btn.configure(bg="#1a2235", fg=FG)
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

    def _manual_load(self):
        p = self._dir_var.get().strip()
        if p:
            self.load_run_dir(Path(p))

    def load_run_dir(self, run_dir: Path):
        """Called automatically after a successful Analyze run, or manually."""
        if not _RESULTS_AVAILABLE:
            return
        self._run_dir = run_dir
        self._dir_var.set(str(run_dir))

        # Load video
        video_path = run_dir / "diagnostic.mp4"
        if video_path.exists():
            ok = self._player.load(video_path)
            status = "video loaded" if ok else "video failed to load"
        else:
            self._player.unload()
            status = "no diagnostic.mp4 found"

        # Load angles CSV
        csv_path = run_dir / "derived" / "angles_sagittal.csv"
        if csv_path.exists():
            ok2 = self._table.load(csv_path)
            status += " | " + ("angles loaded" if ok2 else "angles failed to load")
        else:
            self._table.clear()
            status += " | no angles CSV found"

        self._info_lbl.configure(text=status, fg=FG_DIM)


# ── Analyze tab ───────────────────────────────────────────────────────────────

class AnalyzeTab(tk.Frame):
    def __init__(self, parent, log: LogPane, on_complete=None):
        super().__init__(parent, bg=BG)
        self.log         = log
        self.on_complete = on_complete   # callback(run_dir: Path) on success
        self._build()

    def _build(self):
        card = _frame(self)
        card.pack(fill="x", padx=20, pady=16)

        tk.Label(card, text="ANALYZE VIDEO", bg=PANEL, fg=FG,
                 font=FONT_HEADING).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card,
                 text="MediaPipe pose estimation  +  Butterworth filtering  +  sagittal angle extraction",
                 bg=PANEL, fg=FG_DIM, font=(FF, 8)).pack(anchor="w", padx=16, pady=(0, 10))

        _section_sep(card, "INPUTS")

        self.video_var = tk.StringVar()
        _file_row(card, "Video file", self.video_var,
                  lambda: _browse_file(self.video_var, "Select sagittal video",
                                       [("Video", "*.mp4 *.mov *.avi *.MP4 *.MOV"),
                                        ("All", "*.*")]))

        self.out_var = tk.StringVar(value="runs")
        _file_row(card, "Output directory", self.out_var,
                  lambda: _browse_dir(self.out_var, "Select output root"),
                  hint="run subfolders created automatically")

        _section_sep(card, "OPTIONS")

        self.side_var = tk.StringVar(value="right")
        def _side_w(p):
            f = tk.Frame(p, bg=PANEL)
            f.pack(side="left")
            for v, l in (("right", "Right"), ("left", "Left")):
                tk.Radiobutton(f, text=l, variable=self.side_var, value=v,
                               bg=PANEL, fg=FG, selectcolor="#1c2333",
                               activebackground=PANEL, font=FONT_BODY).pack(side="left", padx=4)
        _option_row(card, "Visible side", _side_w)

        self.height_var = tk.StringVar()
        def _height_w(p):
            _entry(p, self.height_var, width=8).pack(side="left")
            tk.Label(p, text="m  (optional - enables velocity output)",
                     bg=PANEL, fg=FG_DIM, font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Athlete height", _height_w)

        self.cutoff_var = tk.StringVar(value="6.0")
        def _cutoff_w(p):
            _entry(p, self.cutoff_var, width=6).pack(side="left")
            tk.Label(p, text="Hz  (6=walk  10=jog  12=sprint)",
                     bg=PANEL, fg=FG_DIM, font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Butterworth cutoff", _cutoff_w)

        self.max_frames_var = tk.StringVar(value="0")
        def _mf_w(p):
            _entry(p, self.max_frames_var, width=8).pack(side="left")
            tk.Label(p, text="0 = all frames", bg=PANEL, fg=FG_DIM,
                     font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Max frames", _mf_w)

        self.no_diag_var = tk.BooleanVar(value=False)
        def _diag_w(p):
            tk.Checkbutton(p, text="Skip diagnostic video overlay (faster)",
                           variable=self.no_diag_var, bg=PANEL, fg=FG,
                           selectcolor="#1c2333", activebackground=PANEL,
                           font=FONT_BODY).pack(side="left")
        _option_row(card, "Diagnostic video", _diag_w)

        self.mocap_var = tk.StringVar()
        _file_row(card, "MoCap file (opt.)", self.mocap_var,
                  lambda: _browse_file(self.mocap_var, "Select MoCap / SPT reference",
                                       [("MoCap", "*.csv *.txt"), ("All", "*.*")]),
                  hint="runs comparison automatically if provided")

        btn_row = tk.Frame(card, bg=PANEL)
        btn_row.pack(fill="x", padx=16, pady=(14, 16))
        self.run_btn = _button(btn_row, "Run Analysis", self._run, accent=True)
        self.run_btn.pack(side="left")
        self.status_lbl = tk.Label(btn_row, text="", bg=PANEL, fg=FG_DIM,
                                   font=FONT_LABEL)
        self.status_lbl.pack(side="left", padx=12)

    def _run(self):
        video = self.video_var.get().strip()
        if not video:
            self.log.writeln("No video file selected.", "error")
            return

        cmd = ["poetry", "run", "drone-mocap", "analyze",
               "--video", video,
               "--out-dir", self.out_var.get().strip() or "runs",
               "--side", self.side_var.get(),
               "--cutoff-hz", self.cutoff_var.get().strip() or "6.0",
               "--max-frames", self.max_frames_var.get().strip() or "0"]

        h = self.height_var.get().strip()
        if h:
            cmd += ["--athlete-height-m", h]
        if self.no_diag_var.get():
            cmd.append("--no-diagnostic")
        mocap = self.mocap_var.get().strip()
        if mocap:
            cmd += ["--mocap-txt", mocap]

        self.run_btn.configure(state="disabled", text="Running...")
        self.status_lbl.configure(text="")
        self.log.clear()
        self.log.writeln("$ " + " ".join(cmd), "dim")
        self.log.writeln("")

        def worker():
            run_dir_found: Path | None = None
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True, bufsize=1,
                                        encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    # Parse "Done. Outputs in: <path>" to get the run dir
                    m = re.search(r"[Oo]utputs? in[:\s]+(.+)", line.strip())
                    if m:
                        run_dir_found = Path(m.group(1).strip())
                    tag = ("error"   if any(w in line.lower() for w in ("error", "traceback", "exception"))
                           else "warning" if "warn" in line.lower()
                           else "success" if any(w in line.lower() for w in ("done", "outputs in"))
                           else "info")
                    self.log.write(line, tag)
                proc.wait()
                ok = proc.returncode == 0
                self.log.writeln(
                    f"\nFinished (exit {proc.returncode})",
                    "success" if ok else "error",
                )
                def _after():
                    self.run_btn.configure(state="normal", text="Run Analysis")
                    self.status_lbl.configure(
                        text="Done" if ok else f"Failed (code {proc.returncode})",
                        fg=SUCCESS if ok else ERROR_CLR,
                    )
                    if ok and run_dir_found and self.on_complete:
                        self.on_complete(run_dir_found)
                self.run_btn.after(0, _after)
            except Exception as exc:
                self.log.writeln(f"\n{exc}", "error")
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="Run Analysis"),
                    self.status_lbl.configure(text="Error", fg=ERROR_CLR),
                ))

        threading.Thread(target=worker, daemon=True).start()


# ── Evaluate tab ──────────────────────────────────────────────────────────────

class EvaluateTab(tk.Frame):
    def __init__(self, parent, log: LogPane):
        super().__init__(parent, bg=BG)
        self.log = log
        self._build()

    def _build(self):
        card = _frame(self)
        card.pack(fill="x", padx=20, pady=16)

        tk.Label(card, text="EVALUATE vs MoCap", bg=PANEL, fg=FG,
                 font=FONT_HEADING).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card, text="Compare video-derived angles to a markered MoCap / SPT reference",
                 bg=PANEL, fg=FG_DIM, font=(FF, 8)).pack(anchor="w", padx=16, pady=(0, 10))

        _section_sep(card, "INPUTS")

        self.mocap_var = tk.StringVar()
        _file_row(card, "MoCap / SPT file", self.mocap_var,
                  lambda: _browse_file(self.mocap_var, "Select MoCap reference",
                                       [("MoCap", "*.csv *.txt"), ("All", "*.*")]))

        _section_sep(card, "VIDEO ANGLES SOURCE  (choose one)")

        self.run_dir_var = tk.StringVar()
        _file_row(card, "Run directory", self.run_dir_var,
                  lambda: _browse_dir(self.run_dir_var, "Select a runs/YYYYMMDD_HHMMSS folder"),
                  hint="e.g. runs/20260307_143951")

        tk.Label(card, text=" " * 20 + "- or -", bg=PANEL, fg=FG_DIM,
                 font=(FF, 8)).pack(anchor="w", padx=16, pady=1)

        self.angles_file_var = tk.StringVar()
        _file_row(card, "Angles file", self.angles_file_var,
                  lambda: _browse_file(self.angles_file_var, "Select angles file",
                                       [("Parquet/CSV", "*.parquet *.csv"), ("All", "*.*")]),
                  hint=".parquet or .csv")

        _section_sep(card, "OPTIONS")

        self.side_var = tk.StringVar(value="left")
        def _side_w(p):
            f = tk.Frame(p, bg=PANEL)
            f.pack(side="left")
            for v, l in (("right", "Right"), ("left", "Left")):
                tk.Radiobutton(f, text=l, variable=self.side_var, value=v,
                               bg=PANEL, fg=FG, selectcolor="#1c2333",
                               activebackground=PANEL, font=FONT_BODY).pack(side="left", padx=4)
        _option_row(card, "Visible side", _side_w)

        self.out_var = tk.StringVar()
        _file_row(card, "Output dir (opt.)", self.out_var,
                  lambda: _browse_dir(self.out_var, "Select output directory"),
                  hint="defaults to <run_dir>/reports")

        btn_row = tk.Frame(card, bg=PANEL)
        btn_row.pack(fill="x", padx=16, pady=(14, 16))
        self.run_btn = _button(btn_row, "Run Evaluation", self._run, accent=True)
        self.run_btn.pack(side="left")
        self.status_lbl = tk.Label(btn_row, text="", bg=PANEL, fg=FG_DIM,
                                   font=FONT_LABEL)
        self.status_lbl.pack(side="left", padx=12)

    def _run(self):
        mocap      = self.mocap_var.get().strip()
        run_dir    = self.run_dir_var.get().strip()
        angles_f   = self.angles_file_var.get().strip()

        if not mocap:
            self.log.writeln("No MoCap file selected.", "error"); return
        if not run_dir and not angles_f:
            self.log.writeln("Provide either a run directory or an angles file.", "error"); return

        cmd = ["poetry", "run", "drone-mocap", "evaluate",
               "--mocap", mocap, "--side", self.side_var.get()]
        if run_dir:   cmd += ["--run-dir",     run_dir]
        if angles_f:  cmd += ["--angles-file", angles_f]
        out = self.out_var.get().strip()
        if out:       cmd += ["--out-dir", out]

        self.run_btn.configure(state="disabled", text="Running...")
        self.status_lbl.configure(text="")
        self.log.clear()
        self.log.writeln("$ " + " ".join(cmd), "dim")
        self.log.writeln("")

        def worker():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True, bufsize=1,
                                        encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    tag = ("error"   if any(w in line.lower() for w in ("error", "traceback", "exception", "keyerror"))
                           else "warning" if "warn" in line.lower()
                           else "success" if any(w in line.lower() for w in ("done", "outputs written"))
                           else "accent"  if any(w in line.lower() for w in ("rmse", "mae", "pearson"))
                           else "info")
                    self.log.write(line, tag)
                proc.wait()
                ok = proc.returncode == 0
                self.log.writeln(
                    f"\nFinished (exit {proc.returncode})",
                    "success" if ok else "error",
                )
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="Run Evaluation"),
                    self.status_lbl.configure(
                        text="Done" if ok else f"Failed (code {proc.returncode})",
                        fg=SUCCESS if ok else ERROR_CLR,
                    ),
                ))
            except Exception as exc:
                self.log.writeln(f"\n{exc}", "error")
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="Run Evaluation"),
                    self.status_lbl.configure(text="Error", fg=ERROR_CLR),
                ))

        threading.Thread(target=worker, daemon=True).start()


# ── Tab bar ───────────────────────────────────────────────────────────────────

class TabBar(tk.Frame):
    def __init__(self, parent, tabs: list[tuple[str, tk.Frame]], **kw):
        super().__init__(parent, bg=BG, **kw)
        self._tabs   = tabs
        self._active = 0
        self._btns: list[tk.Button] = []

        for i, (label, frame) in enumerate(tabs):
            btn = tk.Button(self, text=label,
                            command=lambda i=i: self.select(i),
                            bg=PANEL if i == 0 else BG,
                            fg=FG   if i == 0 else FG_DIM,
                            relief="flat", font=(FF, 10, "bold"),
                            padx=18, pady=8, bd=0, cursor="hand2")
            btn.pack(side="left")
            self._btns.append(btn)
            ind = tk.Frame(self, bg=ACCENT if i == 0 else BG, height=2)
            ind.pack(side="left", fill="y")
            btn._indicator = ind  # type: ignore[attr-defined]

        tabs[0][1].pack(fill="both", expand=True)

    def select(self, idx: int):
        self._tabs[self._active][1].pack_forget()
        self._btns[self._active].configure(bg=BG, fg=FG_DIM)
        self._btns[self._active]._indicator.configure(bg=BG)  # type: ignore[attr-defined]

        self._active = idx
        self._tabs[idx][1].pack(fill="both", expand=True)
        self._btns[idx].configure(bg=PANEL, fg=FG)
        self._btns[idx]._indicator.configure(bg=ACCENT)        # type: ignore[attr-defined]


# ── Main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroCap")
        self.geometry("980x800")
        self.minsize(800, 620)
        self.configure(bg=BG)
        self.resizable(True, True)
        self._build()

    def _build(self):
        # Title bar
        title_bar = tk.Frame(self, bg="#080c12", height=50)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="  DroCap", bg="#080c12", fg=ACCENT,
                 font=(FF, 13, "bold")).pack(side="left", padx=4, pady=12)
        tk.Label(title_bar, text="  sagittal markerless motion capture",
                 bg="#080c12", fg=FG_DIM, font=(FF, 9)).pack(side="left", pady=16)

        # Log pane (bottom, fixed)
        log = LogPane(self, height=180)
        log.pack(fill="x", side="bottom")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")

        # Status bar (below log)
        status = tk.Frame(self, bg="#080c12", height=22)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        tk.Label(status, text="  Ready", bg="#080c12", fg=FG_DIM,
                 font=(FF, 8)).pack(side="left", pady=3)
        tk.Label(status, text=f"Python {sys.version.split()[0]}  ",
                 bg="#080c12", fg=FG_DIM, font=(FF, 8)).pack(side="right", pady=3)

        # Main content area
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        # Build tabs
        results_tab  = ResultsTab(content)
        analyze_tab  = AnalyzeTab(
            content, log,
            on_complete=lambda run_dir: (
                results_tab.load_run_dir(run_dir),
                tab_bar.select(2),        # switch to Results tab
            ),
        )
        evaluate_tab = EvaluateTab(content, log)

        tab_bar = TabBar(content, [
            ("  ANALYZE  ",  analyze_tab),
            ("  EVALUATE  ", evaluate_tab),
            ("  RESULTS  ",  results_tab),
        ])
        tab_bar.pack(fill="x", side="top")


if __name__ == "__main__":
    app = App()
    app.mainloop()
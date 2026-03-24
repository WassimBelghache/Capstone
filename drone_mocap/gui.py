"""
gui.py — DroCap Desktop Interface
======================================
A tkinter-based desktop GUI for the drone_mocap pipeline.
Replaces the terminal commands:

  Analyze:  poetry run drone-mocap analyze --video ... --side ... --athlete-height-m ...
  Evaluate: poetry run drone-mocap evaluate --mocap ... --run-dir ...

Run with:
  python gui.py
  (from the project root, with the venv active, or via: poetry run python gui.py)
"""
from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, ttk

# ── Colour palette ─────────────────────────────────────────────────────────
BG        = "#0e1117"   # near-black background
PANEL     = "#161b24"   # card background
BORDER    = "#242c3a"   # subtle borders
ACCENT    = "#3b8beb"   # blue accent
ACCENT2   = "#1d6fd8"   # darker blue (hover)
SUCCESS   = "#2ea66a"   # green
WARNING   = "#e0a832"   # amber
ERROR_CLR = "#e05454"   # red
FG        = "#d4dbe8"   # primary text
FG_DIM    = "#6b7a96"   # muted text
FG_LABEL  = "#8fa0bc"   # label text

FONT_FAMILY  = "Consolas"   # monospace — feels like instrument software
FONT_BODY    = (FONT_FAMILY, 10)
FONT_LABEL   = (FONT_FAMILY, 9)
FONT_HEADING = (FONT_FAMILY, 11, "bold")
FONT_MONO    = (FONT_FAMILY, 9)
FONT_TITLE   = (FONT_FAMILY, 14, "bold")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _browse_file(var: tk.StringVar, title: str, filetypes: list) -> None:
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _browse_save_dir(var: tk.StringVar, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


# ── Styled widget factories ──────────────────────────────────────────────────

def _frame(parent, **kw) -> tk.Frame:
    kw.setdefault("bg", PANEL)
    kw.setdefault("highlightbackground", BORDER)
    kw.setdefault("highlightthickness", 1)
    return tk.Frame(parent, **kw)


def _label(parent, text, dim=False, heading=False, **kw) -> tk.Label:
    kw.setdefault("bg", kw.get("bg", PANEL))
    kw.setdefault("fg", FG_DIM if dim else (FG if not heading else FG))
    kw.setdefault("font", FONT_HEADING if heading else FONT_LABEL)
    kw.setdefault("anchor", "w")
    return tk.Label(parent, text=text, **kw)


def _entry(parent, textvariable, width=40, **kw) -> tk.Entry:
    return tk.Entry(
        parent,
        textvariable=textvariable,
        width=width,
        bg="#1c2333",
        fg=FG,
        insertbackground=ACCENT,
        relief="flat",
        font=FONT_BODY,
        highlightbackground=BORDER,
        highlightthickness=1,
        highlightcolor=ACCENT,
        **kw,
    )


def _button(parent, text, command, accent=False, small=False, **kw) -> tk.Button:
    bg = ACCENT if accent else "#1e2a3a"
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=FG if not accent else "#ffffff",
        activebackground=ACCENT2 if accent else "#263347",
        activeforeground="#ffffff",
        relief="flat",
        font=FONT_LABEL if small else FONT_BODY,
        padx=10 if not small else 6,
        pady=4 if not small else 2,
        cursor="hand2",
        **kw,
    )


# ── Section separator ────────────────────────────────────────────────────────

def _section_sep(parent, label: str) -> None:
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill="x", padx=16, pady=(14, 4))
    tk.Label(row, text=label, bg=PANEL, fg=FG_LABEL,
             font=(FONT_FAMILY, 9, "bold")).pack(side="left")
    tk.Frame(row, bg=BORDER, height=1).pack(
        side="left", fill="x", expand=True, padx=(8, 0), pady=6)


# ── File / directory picker row ─────────────────────────────────────────────

def _file_row(parent, label: str, var: tk.StringVar,
              browse_fn, hint: str = "") -> None:
    outer = tk.Frame(parent, bg=PANEL)
    outer.pack(fill="x", padx=16, pady=3)
    tk.Label(outer, text=label, bg=PANEL, fg=FG_LABEL,
             font=FONT_LABEL, width=18, anchor="w").pack(side="left")
    _entry(outer, var, width=38).pack(side="left", padx=(0, 6))
    _button(outer, "Browse", browse_fn, small=True).pack(side="left")
    if hint:
        tk.Label(outer, text=hint, bg=PANEL, fg=FG_DIM,
                 font=(FONT_FAMILY, 8)).pack(side="left", padx=(8, 0))


# ── Option row (label + widget) ──────────────────────────────────────────────

def _option_row(parent, label: str, widget_fn) -> None:
    outer = tk.Frame(parent, bg=PANEL)
    outer.pack(fill="x", padx=16, pady=3)
    tk.Label(outer, text=label, bg=PANEL, fg=FG_LABEL,
             font=FONT_LABEL, width=18, anchor="w").pack(side="left")
    widget_fn(outer)


# ── Log widget ───────────────────────────────────────────────────────────────

class LogPane(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        # Header bar
        hdr = tk.Frame(self, bg="#111827", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  OUTPUT LOG", bg="#111827", fg=FG_DIM,
                 font=(FONT_FAMILY, 8, "bold")).pack(side="left", pady=6)
        _button(hdr, "Clear", self.clear, small=True).pack(
            side="right", padx=8, pady=4)

        # Text area
        self.text = tk.Text(
            self,
            bg="#0a0e16",
            fg=FG,
            font=FONT_MONO,
            relief="flat",
            wrap="word",
            state="disabled",
            padx=10,
            pady=8,
            cursor="arrow",
            selectbackground=ACCENT,
            insertbackground=ACCENT,
        )
        sb = tk.Scrollbar(self, orient="vertical", command=self.text.yview,
                          bg=BG, troughcolor=BG, width=10)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

        # Tag colours
        self.text.tag_config("info",    foreground=FG)
        self.text.tag_config("success", foreground=SUCCESS)
        self.text.tag_config("warning", foreground=WARNING)
        self.text.tag_config("error",   foreground=ERROR_CLR)
        self.text.tag_config("dim",     foreground=FG_DIM)
        self.text.tag_config("accent",  foreground=ACCENT)

    def write(self, text: str, tag: str = "info") -> None:
        self.text.configure(state="normal")
        self.text.insert("end", text, tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def writeln(self, text: str, tag: str = "info") -> None:
        self.write(text + "\n", tag)

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


# ── Analyze tab ──────────────────────────────────────────────────────────────

class AnalyzeTab(tk.Frame):
    def __init__(self, parent, log: LogPane):
        super().__init__(parent, bg=BG)
        self.log = log
        self._build()

    def _build(self) -> None:
        card = _frame(self)
        card.pack(fill="x", padx=20, pady=16)

        tk.Label(card, text="ANALYZE VIDEO", bg=PANEL, fg=FG,
                 font=FONT_HEADING).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card, text="Run MediaPipe pose estimation + Butterworth filtering + sagittal angle extraction",
                 bg=PANEL, fg=FG_DIM, font=(FONT_FAMILY, 8)).pack(anchor="w", padx=16, pady=(0, 10))

        # ── Inputs ──
        _section_sep(card, "INPUTS")

        self.video_var = tk.StringVar()
        _file_row(card, "Video file", self.video_var,
                  lambda: _browse_file(self.video_var, "Select sagittal video",
                                       [("Video", "*.mp4 *.mov *.avi *.MP4 *.MOV"),
                                        ("All", "*.*")]))

        self.out_var = tk.StringVar(value="runs")
        _file_row(card, "Output directory", self.out_var,
                  lambda: _browse_save_dir(self.out_var, "Select output root"),
                  hint="run subfolders created automatically")

        # ── Options ──
        _section_sep(card, "OPTIONS")

        self.side_var = tk.StringVar(value="right")
        def _side_widget(p):
            f = tk.Frame(p, bg=PANEL)
            f.pack(side="left")
            for val, lbl in (("right", "Right"), ("left", "Left")):
                tk.Radiobutton(f, text=lbl, variable=self.side_var, value=val,
                               bg=PANEL, fg=FG, selectcolor="#1c2333",
                               activebackground=PANEL, font=FONT_BODY).pack(side="left", padx=4)
        _option_row(card, "Visible side", _side_widget)

        self.height_var = tk.StringVar()
        def _height_widget(p):
            e = _entry(p, self.height_var, width=8)
            e.pack(side="left")
            tk.Label(p, text="m  (optional — enables velocity output)",
                     bg=PANEL, fg=FG_DIM, font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Athlete height", _height_widget)

        self.cutoff_var = tk.StringVar(value="6.0")
        def _cutoff_widget(p):
            e = _entry(p, self.cutoff_var, width=6)
            e.pack(side="left")
            tk.Label(p, text="Hz  (6=walk · 10=jog · 12=sprint)",
                     bg=PANEL, fg=FG_DIM, font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Butterworth cutoff", _cutoff_widget)

        self.max_frames_var = tk.StringVar(value="0")
        def _mf_widget(p):
            e = _entry(p, self.max_frames_var, width=8)
            e.pack(side="left")
            tk.Label(p, text="0 = all frames",
                     bg=PANEL, fg=FG_DIM, font=FONT_LABEL).pack(side="left", padx=6)
        _option_row(card, "Max frames", _mf_widget)

        self.no_diag_var = tk.BooleanVar(value=False)
        def _diag_widget(p):
            tk.Checkbutton(p, text="Skip diagnostic video overlay (faster)",
                           variable=self.no_diag_var,
                           bg=PANEL, fg=FG, selectcolor="#1c2333",
                           activebackground=PANEL, font=FONT_BODY).pack(side="left")
        _option_row(card, "Diagnostic video", _diag_widget)

        self.mocap_var = tk.StringVar()
        _file_row(card, "MoCap file (opt.)", self.mocap_var,
                  lambda: _browse_file(self.mocap_var, "Select MoCap / SPT reference",
                                       [("MoCap", "*.csv *.txt"), ("All", "*.*")]),
                  hint="runs comparison automatically if provided")

        # ── Run button ──
        btn_row = tk.Frame(card, bg=PANEL)
        btn_row.pack(fill="x", padx=16, pady=(14, 16))
        self.run_btn = _button(btn_row, "▶  Run Analysis", self._run, accent=True)
        self.run_btn.pack(side="left")
        self.status_lbl = tk.Label(btn_row, text="", bg=PANEL, fg=FG_DIM,
                                   font=FONT_LABEL)
        self.status_lbl.pack(side="left", padx=12)

    def _run(self) -> None:
        video = self.video_var.get().strip()
        if not video:
            self.log.writeln("✖  No video file selected.", "error")
            return

        cmd = [
            "poetry", "run", "drone-mocap", "analyze",
            "--video", video,
            "--out-dir", self.out_var.get().strip() or "runs",
            "--side", self.side_var.get(),
            "--cutoff-hz", self.cutoff_var.get().strip() or "6.0",
            "--max-frames", self.max_frames_var.get().strip() or "0",
        ]

        h = self.height_var.get().strip()
        if h:
            cmd += ["--athlete-height-m", h]

        if self.no_diag_var.get():
            cmd.append("--no-diagnostic")

        mocap = self.mocap_var.get().strip()
        if mocap:
            cmd += ["--mocap-txt", mocap]

        self._launch(cmd)

    def _launch(self, cmd: list[str]) -> None:
        self.run_btn.configure(state="disabled", text="⏳ Running…")
        self.status_lbl.configure(text="")
        self.log.clear()
        self.log.writeln("$ " + " ".join(cmd), "dim")
        self.log.writeln("")

        def worker():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    tag = ("error"   if any(w in line.lower() for w in ("error", "traceback", "exception"))
                           else "warning" if any(w in line.lower() for w in ("warning", "warn", "⚠"))
                           else "success" if any(w in line.lower() for w in ("done", "✓", "outputs in"))
                           else "info")
                    self.log.write(line, tag)
                proc.wait()
                ok = proc.returncode == 0
                self.log.writeln(
                    f"\n{'✔  Finished (exit 0)' if ok else f'✖  Exited with code {proc.returncode}'}",
                    "success" if ok else "error",
                )
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="▶  Run Analysis"),
                    self.status_lbl.configure(
                        text="Done ✔" if ok else f"Failed (code {proc.returncode})",
                        fg=SUCCESS if ok else ERROR_CLR,
                    ),
                ))
            except Exception as exc:
                self.log.writeln(f"\n✖  {exc}", "error")
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="▶  Run Analysis"),
                    self.status_lbl.configure(text="Error", fg=ERROR_CLR),
                ))

        threading.Thread(target=worker, daemon=True).start()


# ── Evaluate tab ─────────────────────────────────────────────────────────────

class EvaluateTab(tk.Frame):
    def __init__(self, parent, log: LogPane):
        super().__init__(parent, bg=BG)
        self.log = log
        self._build()

    def _build(self) -> None:
        card = _frame(self)
        card.pack(fill="x", padx=20, pady=16)

        tk.Label(card, text="EVALUATE vs MoCap", bg=PANEL, fg=FG,
                 font=FONT_HEADING).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card, text="Compare video-derived angles to a markered MoCap / SPT reference",
                 bg=PANEL, fg=FG_DIM, font=(FONT_FAMILY, 8)).pack(anchor="w", padx=16, pady=(0, 10))

        _section_sep(card, "INPUTS")

        self.mocap_var = tk.StringVar()
        _file_row(card, "MoCap / SPT file", self.mocap_var,
                  lambda: _browse_file(self.mocap_var, "Select MoCap reference",
                                       [("MoCap", "*.csv *.txt"), ("All", "*.*")]))

        _section_sep(card, "VIDEO ANGLES SOURCE  (choose one)")

        self.run_dir_var = tk.StringVar()
        _file_row(card, "Run directory", self.run_dir_var,
                  lambda: _browse_dir(self.run_dir_var, "Select a runs/YYYYMMDD_HHMMSS folder"),
                  hint="runs/20260307_143951")

        lbl_row = tk.Frame(card, bg=PANEL)
        lbl_row.pack(fill="x", padx=16, pady=1)
        tk.Label(lbl_row, text=" " * 19 + "— or —", bg=PANEL, fg=FG_DIM,
                 font=(FONT_FAMILY, 8)).pack(anchor="w")

        self.angles_file_var = tk.StringVar()
        _file_row(card, "Angles file", self.angles_file_var,
                  lambda: _browse_file(self.angles_file_var, "Select angles file",
                                       [("Parquet/CSV", "*.parquet *.csv"), ("All", "*.*")]),
                  hint=".parquet or .csv")

        _section_sep(card, "OPTIONS")

        self.side_var = tk.StringVar(value="left")
        def _side_widget(p):
            f = tk.Frame(p, bg=PANEL)
            f.pack(side="left")
            for val, lbl in (("right", "Right"), ("left", "Left")):
                tk.Radiobutton(f, text=lbl, variable=self.side_var, value=val,
                               bg=PANEL, fg=FG, selectcolor="#1c2333",
                               activebackground=PANEL, font=FONT_BODY).pack(side="left", padx=4)
        _option_row(card, "Visible side", _side_widget)

        self.out_var = tk.StringVar()
        _file_row(card, "Output dir (opt.)", self.out_var,
                  lambda: _browse_save_dir(self.out_var, "Select output directory"),
                  hint="defaults to <run_dir>/reports")

        # ── Run button ──
        btn_row = tk.Frame(card, bg=PANEL)
        btn_row.pack(fill="x", padx=16, pady=(14, 16))
        self.run_btn = _button(btn_row, "▶  Run Evaluation", self._run, accent=True)
        self.run_btn.pack(side="left")
        self.status_lbl = tk.Label(btn_row, text="", bg=PANEL, fg=FG_DIM,
                                   font=FONT_LABEL)
        self.status_lbl.pack(side="left", padx=12)

    def _run(self) -> None:
        mocap = self.mocap_var.get().strip()
        run_dir = self.run_dir_var.get().strip()
        angles_file = self.angles_file_var.get().strip()

        if not mocap:
            self.log.writeln("✖  No MoCap file selected.", "error")
            return
        if not run_dir and not angles_file:
            self.log.writeln("✖  Provide either a run directory or an angles file.", "error")
            return

        cmd = [
            "poetry", "run", "drone-mocap", "evaluate",
            "--mocap", mocap,
            "--side", self.side_var.get(),
        ]
        if run_dir:
            cmd += ["--run-dir", run_dir]
        if angles_file:
            cmd += ["--angles-file", angles_file]
        out = self.out_var.get().strip()
        if out:
            cmd += ["--out-dir", out]

        self._launch(cmd)

    def _launch(self, cmd: list[str]) -> None:
        self.run_btn.configure(state="disabled", text="⏳ Running…")
        self.status_lbl.configure(text="")
        self.log.clear()
        self.log.writeln("$ " + " ".join(cmd), "dim")
        self.log.writeln("")

        def worker():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    tag = ("error"   if any(w in line.lower() for w in ("error", "traceback", "exception", "keyerror"))
                           else "warning" if any(w in line.lower() for w in ("warning", "warn", "⚠"))
                           else "success" if any(w in line.lower() for w in ("done", "✓", "outputs written", "finished"))
                           else "accent"  if any(w in line.lower() for w in ("rmse", "mae", "pearson"))
                           else "info")
                    self.log.write(line, tag)
                proc.wait()
                ok = proc.returncode == 0
                self.log.writeln(
                    f"\n{'✔  Finished (exit 0)' if ok else f'✖  Exited with code {proc.returncode}'}",
                    "success" if ok else "error",
                )
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="▶  Run Evaluation"),
                    self.status_lbl.configure(
                        text="Done ✔" if ok else f"Failed (code {proc.returncode})",
                        fg=SUCCESS if ok else ERROR_CLR,
                    ),
                ))
            except Exception as exc:
                self.log.writeln(f"\n✖  {exc}", "error")
                self.run_btn.after(0, lambda: (
                    self.run_btn.configure(state="normal", text="▶  Run Evaluation"),
                    self.status_lbl.configure(text="Error", fg=ERROR_CLR),
                ))

        threading.Thread(target=worker, daemon=True).start()


# ── Custom tab bar ────────────────────────────────────────────────────────────

class TabBar(tk.Frame):
    def __init__(self, parent, tabs: list[tuple[str, tk.Frame]], **kw):
        super().__init__(parent, bg=BG, **kw)
        self._tabs   = tabs
        self._active = 0
        self._btns: list[tk.Button] = []

        for i, (label, frame) in enumerate(tabs):
            btn = tk.Button(
                self,
                text=label,
                command=lambda i=i: self._select(i),
                bg=PANEL if i == 0 else BG,
                fg=FG   if i == 0 else FG_DIM,
                relief="flat",
                font=(FONT_FAMILY, 10, "bold"),
                padx=18,
                pady=8,
                bd=0,
                cursor="hand2",
            )
            btn.pack(side="left")
            self._btns.append(btn)
            # Active indicator bar
            ind = tk.Frame(self, bg=ACCENT if i == 0 else BG, height=2)
            ind.pack(side="left", fill="y")
            btn._indicator = ind   # type: ignore[attr-defined]

        # Show first tab
        tabs[0][1].pack(fill="both", expand=True)

    def _select(self, idx: int) -> None:
        self._tabs[self._active][1].pack_forget()
        self._btns[self._active].configure(bg=BG, fg=FG_DIM)
        self._btns[self._active]._indicator.configure(bg=BG)   # type: ignore[attr-defined]

        self._active = idx
        self._tabs[idx][1].pack(fill="both", expand=True)
        self._btns[idx].configure(bg=PANEL, fg=FG)
        self._btns[idx]._indicator.configure(bg=ACCENT)         # type: ignore[attr-defined]


# ── Main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroCap")
        self.geometry("860x720")
        self.minsize(720, 580)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._build()

    def _build(self) -> None:
        # ── Title bar ──
        title_bar = tk.Frame(self, bg="#080c12", height=50)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="  DROCAP",
            bg="#080c12",
            fg=ACCENT,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(side="left", padx=4, pady=12)

        tk.Label(
            title_bar,
            text="  sagittal markerless motion capture",
            bg="#080c12",
            fg=FG_DIM,
            font=(FONT_FAMILY, 9),
        ).pack(side="left", pady=16)

        # ── Tab container (upper) ──
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        # Left pane: tabs
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # Shared log pane (bottom)
        log = LogPane(self, height=220)
        log.pack(fill="x", side="bottom")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")

        # Build tabs
        analyze_tab  = AnalyzeTab(left, log)
        evaluate_tab = EvaluateTab(left, log)

        tab_bar = TabBar(left, [
            ("  ANALYZE  ", analyze_tab),
            ("  EVALUATE  ", evaluate_tab),
        ])
        tab_bar.pack(fill="x", side="top")

        # Status bar
        status = tk.Frame(self, bg="#080c12", height=22)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        tk.Label(
            status,
            text="  Ready",
            bg="#080c12",
            fg=FG_DIM,
            font=(FONT_FAMILY, 8),
        ).pack(side="left", pady=3)
        tk.Label(
            status,
            text=f"Python {sys.version.split()[0]}  ",
            bg="#080c12",
            fg=FG_DIM,
            font=(FONT_FAMILY, 8),
        ).pack(side="right", pady=3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
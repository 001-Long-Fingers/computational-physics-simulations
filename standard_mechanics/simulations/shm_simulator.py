import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Palette
BG           = "#f5f4f0"
PANEL_BG     = "#ededea"
CARD_BG      = "#f5f4f0"
BORDER       = "#d3d1c7"
TEXT_PRI     = "#2c2c2a"
TEXT_SEC     = "#888780"
ACCENT       = "#3266ad"
RED_ACC      = "#c0392b"
ORANGE       = "#e67e22"
GREEN        = "#27ae60"
PURPLE       = "#8e44ad"
BADGE_RUN    = "#d4edda"
BADGE_RUN_FG = "#155724"
BADGE_IDL_FG = "#888780"


# Physics
def build_sim(m, k, x0, v0, tmax, zeta, dt=0.01):
    omega   = np.sqrt(k / m)
    omega_d = omega * np.sqrt(max(0.0, 1 - zeta ** 2))
    t = np.arange(0, tmax + dt, dt)

    if zeta == 0:
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    elif zeta < 1:
        A, B   = x0, (v0 + zeta * omega * x0) / omega_d
        decay  = np.exp(-zeta * omega * t)
        x = decay * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        v = (-zeta * omega * x
             + decay * (-A * omega_d * np.sin(omega_d * t)
                        + B * omega_d * np.cos(omega_d * t)))
    else:
        sq = np.sqrt(zeta ** 2 - 1)
        r1, r2 = -omega * (zeta - sq), -omega * (zeta + sq)
        A = (v0 - r2 * x0) / (r1 - r2);  B = x0 - A
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        v = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)

    return dict(t=t, x=x, v=v,
                ke=0.5 * m * v ** 2, pe=0.5 * k * x ** 2,
                omega=omega, T=2 * np.pi / omega,
                amp=np.sqrt(x0 ** 2 + (v0 / omega) ** 2))


class SHMApp(tk.Tk):
    TABS = [("Displacement", "x"), ("Velocity", "v"),
            ("Energy", "e"), ("Phase", "p")]

    def __init__(self):
        super().__init__()
        self.title("SHM Simulator")
        self.configure(bg=BG)
        self.minsize(960, 600)
        self.geometry("1100x680")

        self._anim    = None
        self._sim     = None
        self._cur_tab = "x"
        self._vars    = {}

        self._build_ui()
        self._update_stats()

# Layout
    def _build_ui(self):
        # root grid: 1 row, 2 columns
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, minsize=270)
        self.columnconfigure(1, weight=1)

        self._make_left_panel()
        self._make_right_panel()

    # Left panel
    def _make_left_panel(self):
        lf = tk.Frame(self, bg=PANEL_BG,
                      highlightthickness=1, highlightbackground=BORDER)
        lf.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        lf.columnconfigure(0, weight=1)

        tk.Label(lf, text="PARAMETERS", bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        params = [
            ("Mass (kg)",                    "mass",  0.1, 5.0,  1.0,  0.1,  1),
            ("Spring constant k (N/m)",      "k",     1,   200,  50,   1,    0),
            ("Initial displacement x₀ (m)",  "x0",  -2.0, 2.0,  0.1,  0.05, 2),
            ("Initial velocity v₀ (m/s)",    "v0",  -5.0, 5.0,  0.0,  0.1,  1),
            ("Simulation time (s)",          "tmax",  1,   20,   5,    0.5,  0),
            ("Damping ζ  (0 = undamped)",    "damp",  0.0, 2.0,  0.0,  0.01, 2),
        ]
        for r, (lbl, key, lo, hi, default, step, dec) in enumerate(params, start=1):
            self._add_slider(lf, r, lbl, key, lo, hi, default, dec)

        run_btn = tk.Button(lf, text="▶  Run simulation",
                            bg=TEXT_PRI, fg="white",
                            activebackground="#444", activeforeground="white",
                            font=("Helvetica", 12), relief="flat",
                            cursor="hand2", pady=8, command=self._run)
        run_btn.grid(row=8, column=0, sticky="ew", padx=16, pady=(16, 4))

        stop_btn = tk.Button(lf, text="◼  Stop",
                             bg=PANEL_BG, fg=TEXT_SEC,
                             activebackground=BORDER,
                             font=("Helvetica", 11), relief="flat",
                             highlightthickness=1, highlightbackground=BORDER,
                             cursor="hand2", pady=6, command=self._stop)
        stop_btn.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 16))

    def _add_slider(self, parent, row, label, key, lo, hi, default, dec):
        outer = tk.Frame(parent, bg=PANEL_BG)
        outer.grid(row=row, column=0, sticky="ew", padx=16, pady=3)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text=label, bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 10)).grid(row=0, column=0, columnspan=2, sticky="w")

        var = tk.DoubleVar(value=default)
        self._vars[key] = (var, dec)

        val_lbl = tk.Label(outer, text=f"{default:.{dec}f}",
                           bg=PANEL_BG, fg=TEXT_PRI,
                           font=("Helvetica", 10, "bold"), width=6, anchor="e")
        val_lbl.grid(row=1, column=1, sticky="e")

        def on_change(_, v=var, l=val_lbl, d=dec):
            l.config(text=f"{v.get():.{d}f}")
            self._update_stats()

        sl = ttk.Scale(outer, from_=lo, to=hi, variable=var,
                       orient="horizontal", command=on_change)
        sl.grid(row=1, column=0, sticky="ew", padx=(0, 6))

    # Right panel
    def _make_right_panel(self):
        rf = tk.Frame(self, bg=BG)
        rf.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        # rf grid: row0=cards, row1=tabs+badge, row2=canvas (weight=1)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(2, weight=1)

        # stat cards
        cards = tk.Frame(rf, bg=BG)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for c in range(3):
            cards.columnconfigure(c, weight=1, uniform="sc")

        self._lbl_omega = self._stat_card(cards, 0, "Angular freq ω", "—")
        self._lbl_T     = self._stat_card(cards, 1, "Period T",        "—")
        self._lbl_A     = self._stat_card(cards, 2, "Amplitude A",     "—")

        # tabs + badge
        tb = tk.Frame(rf, bg=BG)
        tb.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        tb.columnconfigure(0, weight=1)

        tab_inner = tk.Frame(tb, bg=BG)
        tab_inner.grid(row=0, column=0, sticky="w")
        self._tab_btns = {}
        for lbl, key in self.TABS:
            btn = tk.Button(tab_inner, text=lbl,
                            font=("Helvetica", 10), relief="flat",
                            cursor="hand2", padx=10, pady=4,
                            command=lambda k=key: self._set_tab(k))
            btn.pack(side="left", padx=(0, 4))
            self._tab_btns[key] = btn
        self._refresh_tabs()

        self._badge = tk.Label(tb, text="idle", bg=CARD_BG, fg=TEXT_SEC,
                               font=("Helvetica", 9), padx=8, pady=3,
                               highlightthickness=1, highlightbackground=BORDER)
        self._badge.grid(row=0, column=1, sticky="e")

        # matplotlib canvas
        self._fig = Figure(facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._style_axes()

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=rf)
        widget = self._mpl_canvas.get_tk_widget()
        widget.grid(row=2, column=0, sticky="nsew")

    def _stat_card(self, parent, col, label_text, value_text):
        pad = (0, 6) if col == 0 else (6, 0) if col == 2 else (3, 3)
        card = tk.Frame(parent, bg=CARD_BG,
                        highlightthickness=1, highlightbackground=BORDER,
                        padx=12, pady=8)
        card.grid(row=0, column=col, sticky="ew", padx=pad)
        tk.Label(card, text=label_text, bg=CARD_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9)).pack(anchor="w")
        lbl = tk.Label(card, text=value_text, bg=CARD_BG, fg=TEXT_PRI,
                       font=("Helvetica", 16, "bold"))
        lbl.pack(anchor="w")
        return lbl

    # Axes
    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(BG)
        self._fig.patch.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.grid(True, color=BORDER, linewidth=0.5)
        self._fig.tight_layout(pad=1.6)

    # Stats
    def _update_stats(self):
        m  = self._vars["mass"][0].get()
        k  = self._vars["k"][0].get()
        x0 = self._vars["x0"][0].get()
        v0 = self._vars["v0"][0].get()
        omega = np.sqrt(k / m)
        T     = 2 * np.pi / omega
        amp   = np.sqrt(x0 ** 2 + (v0 / omega) ** 2)
        self._lbl_omega.config(text=f"{omega:.2f} rad/s")
        self._lbl_T    .config(text=f"{T:.3f} s")
        self._lbl_A    .config(text=f"{amp:.3f} m")

    # Tabs
    def _set_tab(self, key):
        self._cur_tab = key
        self._refresh_tabs()
        if self._sim:
            self._draw_partial(self._sim, len(self._sim["t"]))

    def _refresh_tabs(self):
        for k, btn in self._tab_btns.items():
            active = k == self._cur_tab
            btn.config(bg=TEXT_PRI if active else BG,
                       fg="white"   if active else TEXT_SEC,
                       highlightthickness=1,
                       highlightbackground=TEXT_PRI if active else BORDER)

    # Run / Stop
    def _run(self):
        self._stop()
        p = {k: v.get() for k, (v, _) in self._vars.items()}
        self._sim = build_sim(p["mass"], p["k"], p["x0"],
                              p["v0"], p["tmax"], p["damp"])
        self._update_stats()
        self._badge.config(text="running", bg=BADGE_RUN, fg=BADGE_RUN_FG)

        total = len(self._sim["t"])
        step  = max(1, total // 300)

        def update(i):
            n = min((i + 1) * step, total)
            self._draw_partial(self._sim, n)
            if n >= total:
                self._badge.config(text="done", bg=CARD_BG, fg=BADGE_IDL_FG)

        self._anim = animation.FuncAnimation(
            self._fig, update,
            frames=total // step + 1,
            interval=16, blit=False, repeat=False)
        self._mpl_canvas.draw()

    def _stop(self):
        if self._anim:
            self._anim.event_source.stop()
            self._anim = None
        self._badge.config(text="idle", bg=CARD_BG, fg=BADGE_IDL_FG)

    # Draw
    def _draw_partial(self, sim, n):
        self._ax.cla()
        self._style_axes()
        t = sim["t"][:n]

        if self._cur_tab == "x":
            self._ax.plot(t, sim["x"][:n], color=ACCENT, lw=1.8)
            self._ax.fill_between(t, sim["x"][:n], color=ACCENT, alpha=0.08)
            self._ax.set_xlabel("time (s)");  self._ax.set_ylabel("displacement (m)")
            self._ax.set_xlim(0, sim["t"][-1])

        elif self._cur_tab == "v":
            self._ax.plot(t, sim["v"][:n], color=RED_ACC, lw=1.8)
            self._ax.fill_between(t, sim["v"][:n], color=RED_ACC, alpha=0.08)
            self._ax.set_xlabel("time (s)");  self._ax.set_ylabel("velocity (m/s)")
            self._ax.set_xlim(0, sim["t"][-1])

        elif self._cur_tab == "e":
            ke, pe = sim["ke"][:n], sim["pe"][:n]
            self._ax.plot(t, ke,      color=ACCENT,  lw=1.5, label="KE")
            self._ax.plot(t, pe,      color=ORANGE,  lw=1.5, label="PE")
            self._ax.plot(t, ke + pe, color=GREEN,   lw=1.5, ls="--", label="Total E")
            self._ax.set_xlabel("time (s)");  self._ax.set_ylabel("energy (J)")
            self._ax.set_xlim(0, sim["t"][-1])
            self._ax.legend(fontsize=8, facecolor=PANEL_BG,
                            edgecolor=BORDER, labelcolor=TEXT_PRI)

        elif self._cur_tab == "p":
            self._ax.plot(sim["x"][:n], sim["v"][:n], color=PURPLE, lw=1.5)
            self._ax.set_xlabel("x (m)");  self._ax.set_ylabel("v (m/s)")

        self._mpl_canvas.draw_idle()


# Entry point
if __name__ == "__main__":
    app = SHMApp()
    app.mainloop()

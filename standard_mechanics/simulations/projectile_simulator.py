import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Palette ───────────────────────────────────────────────────────────────────
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
TRAIL_GRAY   = "#b4b2a9"
BADGE_RUN    = "#d4edda"
BADGE_RUN_FG = "#155724"
BADGE_IDL_FG = "#888780"

G = 9.81


# Physics
def build_sim(mass, v0, angle_deg, b, y0_init, t_max):
    angle = np.radians(angle_deg)
    vx0   = v0 * np.cos(angle)
    vy0   = v0 * np.sin(angle)

    def equations(t, state):
        _, _, vx, vy = state
        ax = -(b / mass) * vx
        ay = -G - (b / mass) * vy
        return [vx, vy, ax, ay]

    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal  = True
    hit_ground.direction = -1

    sol = solve_ivp(equations, (0, t_max), [0.0, y0_init, vx0, vy0],
                    t_eval=np.linspace(0, t_max, 600),
                    events=hit_ground, rtol=1e-9, atol=1e-12)

    xs  = np.array(sol.y[0], dtype=float)
    ys  = np.array(sol.y[1], dtype=float)
    vxs = np.array(sol.y[2], dtype=float)
    vys = np.array(sol.y[3], dtype=float)
    ts  = np.array(sol.t,    dtype=float)

    mask = ys >= -0.01
    xs, ys, vxs, vys, ts = xs[mask], ys[mask], vxs[mask], vys[mask], ts[mask]
    ys = np.clip(ys, 0, None)

    speeds = np.sqrt(vxs**2 + vys**2)
    ke     = 0.5 * mass * speeds**2
    pe     = mass * G * ys

    return dict(x=xs, y=ys, vx=vxs, vy=vys, t=ts,
                ke=ke, pe=pe, speeds=speeds,
                max_range=float(xs[-1]) if len(xs) else 0.0,
                max_height=float(np.max(ys)) if len(ys) else 0.0,
                flight_t=float(ts[-1]) if len(ts) else 0.0)


# App
class ProjectileApp:
    TABS = [("Trajectory", "traj"), ("Speed", "spd"),
            ("Energy",     "e"),    ("Height", "ht")]

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Projectile Simulator")
        root.configure(bg=BG)
        root.minsize(980, 620)
        root.geometry("1140x700")

        self._anim    = None
        self._sim     = None
        self._cur_tab = "traj"
        self._vars    = {}

        self._build_ui()
        self._update_stats()

    # Layout
    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, minsize=280)
        self.root.columnconfigure(1, weight=1)
        self._make_left_panel()
        self._make_right_panel()

    # Left panel
    def _make_left_panel(self):
        lf = tk.Frame(self.root, bg=PANEL_BG,
                      highlightthickness=1, highlightbackground=BORDER)
        lf.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        lf.columnconfigure(0, weight=1)

        tk.Label(lf, text="PARAMETERS", bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        params = [
            ("Mass (kg)",                  "mass",  0.1,  50.0,  10.0, 1),
            ("Initial speed v₀ (m/s)",     "v0",    1.0,  100.0, 20.0, 1),
            ("Launch angle θ (°)",         "angle", 1.0,  89.0,  45.0, 1),
            ("Drag coefficient b (kg/s)",  "drag",  0.0,  5.0,   0.0,  2),
            ("Initial height y₀ (m)",      "y0",    0.0,  100.0, 0.0,  1),
            ("Simulation time (s)",        "tmax",  1.0,  30.0,  10.0, 1),
        ]
        for r, (lbl, key, lo, hi, default, dec) in enumerate(params, start=1):
            self._add_slider(lf, r, lbl, key, lo, hi, default, dec)

        tk.Button(lf, text="▶  Launch",
                  bg=TEXT_PRI, fg="white",
                  activebackground="#444", activeforeground="white",
                  font=("Helvetica", 12), relief="flat",
                  cursor="hand2", pady=8,
                  command=self._run).grid(
            row=8, column=0, sticky="ew", padx=16, pady=(16, 4))

        tk.Button(lf, text="◼  Stop",
                  bg=PANEL_BG, fg=TEXT_SEC,
                  activebackground=BORDER,
                  font=("Helvetica", 11), relief="flat",
                  highlightthickness=1, highlightbackground=BORDER,
                  cursor="hand2", pady=6,
                  command=self._stop).grid(
            row=9, column=0, sticky="ew", padx=16, pady=(0, 16))

    def _add_slider(self, parent, row, label, key, lo, hi, default, dec):
        outer = tk.Frame(parent, bg=PANEL_BG)
        outer.grid(row=row, column=0, sticky="ew", padx=16, pady=3)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text=label, bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 10)).grid(row=0, column=0,
                                              columnspan=2, sticky="w")

        var = tk.DoubleVar(value=default)
        self._vars[key] = (var, dec)

        val_lbl = tk.Label(outer, text=f"{default:.{dec}f}",
                           bg=PANEL_BG, fg=TEXT_PRI,
                           font=("Helvetica", 10, "bold"), width=6, anchor="e")
        val_lbl.grid(row=1, column=1, sticky="e")

        def on_change(_, v=var, l=val_lbl, d=dec):
            l.config(text=f"{v.get():.{d}f}")
            self._update_stats()

        ttk.Scale(outer, from_=lo, to=hi, variable=var,
                  orient="horizontal",
                  command=on_change).grid(row=1, column=0,
                                          sticky="ew", padx=(0, 6))

    # Right panel
    def _make_right_panel(self):
        rf = tk.Frame(self.root, bg=BG)
        rf.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(2, weight=1)

        # stat cards
        cards = tk.Frame(rf, bg=BG)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for c in range(4):
            cards.columnconfigure(c, weight=1, uniform="sc")

        self._lbl_range = self._stat_card(cards, 0, "Range",        "—")
        self._lbl_hmax  = self._stat_card(cards, 1, "Max height",   "—")
        self._lbl_tof   = self._stat_card(cards, 2, "Flight time",  "—")
        self._lbl_vmax  = self._stat_card(cards, 3, "Launch speed", "—")

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
        self._mpl_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

    def _stat_card(self, parent, col, label_text, value_text):
        pads = [(0, 4), (4, 4), (4, 4), (4, 0)]
        card = tk.Frame(parent, bg=CARD_BG,
                        highlightthickness=1, highlightbackground=BORDER,
                        padx=10, pady=8)
        card.grid(row=0, column=col, sticky="ew", padx=pads[col])
        tk.Label(card, text=label_text, bg=CARD_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9)).pack(anchor="w")
        lbl = tk.Label(card, text=value_text, bg=CARD_BG, fg=TEXT_PRI,
                       font=("Helvetica", 15, "bold"))
        lbl.pack(anchor="w")
        return lbl

    # Axes style
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

    # Live stats
    def _update_stats(self):
        v0    = self._vars["v0"][0].get()
        angle = np.radians(self._vars["angle"][0].get())
        y0    = self._vars["y0"][0].get()
        vx0   = v0 * np.cos(angle)
        vy0   = v0 * np.sin(angle)
        t_f   = (vy0 + np.sqrt(max(0, vy0**2 + 2 * G * y0))) / G
        h_max = y0 + vy0**2 / (2 * G)
        rng   = vx0 * t_f
        self._lbl_range.config(text=f"{rng:.1f} m")
        self._lbl_hmax .config(text=f"{h_max:.1f} m")
        self._lbl_tof  .config(text=f"{t_f:.2f} s")
        self._lbl_vmax .config(text=f"{v0:.1f} m/s")

    # Tabs
    def _set_tab(self, key):
        self._cur_tab = key
        self._refresh_tabs()
        if self._sim:
            self._draw_partial(len(self._sim["x"]))

    def _refresh_tabs(self):
        for k, btn in self._tab_btns.items():
            active = k == self._cur_tab
            btn.config(bg=TEXT_PRI if active else BG,
                       fg="white"  if active else TEXT_SEC,
                       highlightthickness=1,
                       highlightbackground=TEXT_PRI if active else BORDER)

    # Run / Stop
    def _run(self):
        self._stop()
        p = {k: v.get() for k, (v, _) in self._vars.items()}
        self._sim = build_sim(p["mass"], p["v0"], p["angle"],
                              p["drag"], p["y0"], p["tmax"])
        s = self._sim
        self._lbl_range.config(text=f"{s['max_range']:.1f} m")
        self._lbl_hmax .config(text=f"{s['max_height']:.1f} m")
        self._lbl_tof  .config(text=f"{s['flight_t']:.2f} s")
        self._lbl_vmax .config(text=f"{p['v0']:.1f} m/s")
        self._badge.config(text="running", bg=BADGE_RUN, fg=BADGE_RUN_FG)

        total = len(s["x"])
        step  = max(1, total // 300)

        def update(i):
            n = min((i + 1) * step, total)
            self._draw_partial(n)
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
    def _draw_partial(self, n):
        s  = self._sim
        ax = self._ax
        ax.cla()
        self._style_axes()

        if self._cur_tab == "traj":
            ax.plot(s["x"], s["y"], color=TRAIL_GRAY, lw=1, ls="--")
            ax.plot(s["x"][:n], s["y"][:n], color=ACCENT, lw=1.8)
            ax.fill_between(s["x"][:n], s["y"][:n], color=ACCENT, alpha=0.07)
            if n > 0:
                ax.plot(s["x"][n-1], s["y"][n-1], 'o',
                        color=RED_ACC, ms=7, zorder=5)
            ax.set_xlabel("x position (m)")
            ax.set_ylabel("y position (m)")
            ax.set_xlim(0, max(float(s["x"].max()) * 1.08, 1))
            ax.set_ylim(0, max(float(s["y"].max()) * 1.2,  1))

        elif self._cur_tab == "spd":
            ax.plot(s["t"][:n], s["speeds"][:n], color=RED_ACC, lw=1.8)
            ax.fill_between(s["t"][:n], s["speeds"][:n], color=RED_ACC, alpha=0.08)
            ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)")
            ax.set_xlim(0, float(s["t"][-1]))

        elif self._cur_tab == "e":
            t = s["t"][:n]
            ax.plot(t, s["ke"][:n],              color=ACCENT, lw=1.5, label="KE")
            ax.plot(t, s["pe"][:n],              color=ORANGE, lw=1.5, label="PE")
            ax.plot(t, s["ke"][:n]+s["pe"][:n], color=GREEN,  lw=1.5,
                    ls="--", label="Total E")
            ax.set_xlabel("time (s)"); ax.set_ylabel("energy (J)")
            ax.set_xlim(0, float(s["t"][-1]))
            ax.legend(fontsize=8, facecolor=PANEL_BG,
                      edgecolor=BORDER, labelcolor=TEXT_PRI)

        elif self._cur_tab == "ht":
            ax.plot(s["t"][:n], s["y"][:n], color=GREEN, lw=1.8)
            ax.fill_between(s["t"][:n], s["y"][:n], color=GREEN, alpha=0.08)
            ax.set_xlabel("time (s)"); ax.set_ylabel("height (m)")
            ax.set_xlim(0, float(s["t"][-1]))
            ax.set_ylim(0, max(float(s["y"].max()) * 1.2, 1))

        self._mpl_canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app  = ProjectileApp(root)
    root.mainloop()

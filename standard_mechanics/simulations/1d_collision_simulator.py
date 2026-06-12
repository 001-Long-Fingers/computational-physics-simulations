import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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

BLOCK_SIZE = 1.0
DT         = 0.01
T_MAX      = 10.0

def build_sim(m1, m2, u1, u2, e):
    # Final velocities via coefficient of restitution
    v1 = ((m1 - e * m2) * u1 + (1 + e) * m2 * u2) / (m1 + m2)
    v2 = ((m2 - e * m1) * u2 + (1 + e) * m1 * u1) / (m1 + m2)

    # Energy
    Ti1, Ti2   = 0.5 * m1 * u1**2, 0.5 * m2 * u2**2
    Tf1, Tf2   = 0.5 * m1 * v1**2, 0.5 * m2 * v2**2
    Ti, Tf     = Ti1 + Ti2, Tf1 + Tf2

    # Momentum
    Pi, Pf     = m1 * u1 + m2 * u2, m1 * v1 + m2 * v2

    # Collision type
    if np.isclose(Ti, Tf, atol=1e-5):
        ctype = "Elastic"
    elif e == 0:
        ctype = "Perfectly Inelastic"
    else:
        ctype = "Inelastic"

    x1, x2 = -8.0, 8.0
    cur_v1, cur_v2 = u1, u2
    collision_done = False
    collision_frame = None

    times, pos1, pos2 = [], [], []
    ke1_arr, ke2_arr  = [], []
    p1_arr,  p2_arr   = [], []

    for i, t in enumerate(np.arange(0, T_MAX, DT)):
        times.append(t)
        pos1.append(x1)
        pos2.append(x2)
        ke1_arr.append(0.5 * m1 * cur_v1**2)
        ke2_arr.append(0.5 * m2 * cur_v2**2)
        p1_arr.append(m1 * cur_v1)
        p2_arr.append(m2 * cur_v2)

        if not collision_done and (x2 - x1) <= BLOCK_SIZE:
            cur_v1, cur_v2 = v1, v2
            collision_done = True
            collision_frame = i

        x1 += cur_v1 * DT
        x2 += cur_v2 * DT

    times    = np.array(times)
    pos1     = np.array(pos1)
    pos2     = np.array(pos2)
    ke1_arr  = np.array(ke1_arr)
    ke2_arr  = np.array(ke2_arr)
    p1_arr   = np.array(p1_arr)
    p2_arr   = np.array(p2_arr)

    return dict(
        times=times, pos1=pos1, pos2=pos2,
        ke1=ke1_arr, ke2=ke2_arr,
        ke_total=ke1_arr + ke2_arr,
        p1=p1_arr, p2=p2_arr,
        p_total=p1_arr + p2_arr,
        v1=v1, v2=v2, u1=u1, u2=u2,
        Ti=Ti, Tf=Tf, Pi=Pi, Pf=Pf,
        Ti1=Ti1, Ti2=Ti2, Tf1=Tf1, Tf2=Tf2,
        delta_T=Tf - Ti, delta_P=Pf - Pi,
        ctype=ctype,
        collision_frame=collision_frame,
    )

class CollisionApp:
    TABS = [("Animation", "anim"), ("Energy", "e"),
            ("Momentum",  "p"),    ("Displacement", "disp")]

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("1D Collision Simulator")
        root.configure(bg=BG)
        root.minsize(980, 640)
        root.geometry("1160x720")

        self._anim    = None
        self._sim     = None
        self._cur_tab = "anim"
        self._vars    = {}

        self._build_ui()
        self._update_stats()

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, minsize=285)
        self.root.columnconfigure(1, weight=1)
        self._make_left_panel()
        self._make_right_panel()

    def _make_left_panel(self):
        lf = tk.Frame(self.root, bg=PANEL_BG,
                      highlightthickness=1, highlightbackground=BORDER)
        lf.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        lf.columnconfigure(0, weight=1)

        tk.Label(lf, text="PARAMETERS", bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        params = [
            ("Mass of Body 1  m₁ (kg)",         "m1",  0.1, 20.0,  5.0,  1),
            ("Mass of Body 2  m₂ (kg)",         "m2",  0.1, 20.0,  5.0,  1),
            ("Initial velocity u₁ (m/s)",        "u1", -15.0, 15.0,  5.0,  1),
            ("Initial velocity u₂ (m/s)",        "u2", -15.0, 15.0, -3.0,  1),
            ("Coeff. of restitution  e",         "e",   0.0,  1.0,   1.0,  2),
        ]
        for r, (lbl, key, lo, hi, default, dec) in enumerate(params, start=1):
            self._add_slider(lf, r, lbl, key, lo, hi, default, dec)

        tk.Button(lf, text="▶  Simulate",
                  bg=TEXT_PRI, fg="white",
                  activebackground="#444", activeforeground="white",
                  font=("Helvetica", 12), relief="flat",
                  cursor="hand2", pady=8,
                  command=self._run).grid(
            row=7, column=0, sticky="ew", padx=16, pady=(16, 4))

        tk.Button(lf, text="◼  Stop",
                  bg=PANEL_BG, fg=TEXT_SEC,
                  activebackground=BORDER,
                  font=("Helvetica", 11), relief="flat",
                  highlightthickness=1, highlightbackground=BORDER,
                  cursor="hand2", pady=6,
                  command=self._stop).grid(
            row=8, column=0, sticky="ew", padx=16, pady=(0, 10))
        sep = tk.Frame(lf, bg=BORDER, height=1)
        sep.grid(row=9, column=0, sticky="ew", padx=16, pady=(4, 8))

        tk.Label(lf, text="RESULTS", bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 9, "bold")).grid(
            row=10, column=0, sticky="w", padx=16, pady=(0, 4))

        self._results_box = tk.Text(
            lf, height=11, width=28,
            bg=BG, fg=TEXT_PRI,
            font=("Courier", 9),
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            state="disabled", wrap="none",
            padx=8, pady=6)
        self._results_box.grid(row=11, column=0, sticky="ew",
                               padx=16, pady=(0, 16))

    def _add_slider(self, parent, row, label, key, lo, hi, default, dec):
        outer = tk.Frame(parent, bg=PANEL_BG)
        outer.grid(row=row, column=0, sticky="ew", padx=16, pady=3)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text=label, bg=PANEL_BG, fg=TEXT_SEC,
                 font=("Helvetica", 10)).grid(
            row=0, column=0, columnspan=2, sticky="w")

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
                  command=on_change).grid(
            row=1, column=0, sticky="ew", padx=(0, 6))

    def _make_right_panel(self):
        rf = tk.Frame(self.root, bg=BG)
        rf.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(2, weight=1)
        cards = tk.Frame(rf, bg=BG)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for c in range(4):
            cards.columnconfigure(c, weight=1, uniform="sc")

        self._lbl_v1    = self._stat_card(cards, 0, "v₁ after",   "—")
        self._lbl_v2    = self._stat_card(cards, 1, "v₂ after",   "—")
        self._lbl_dT    = self._stat_card(cards, 2, "ΔKE",        "—")
        self._lbl_ctype = self._stat_card(cards, 3, "Type",       "—")

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

        self._fig = Figure(facecolor=BG)
        self._ax  = None

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

    def _style_ax(self, ax, yticks=True):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.grid(True, color=BORDER, linewidth=0.5)
        if not yticks:
            ax.set_yticks([])

    def _clear_fig(self):
        self._fig.clf()
        self._fig.patch.set_facecolor(BG)

    def _update_stats(self):
        try:
            m1 = self._vars["m1"][0].get()
            m2 = self._vars["m2"][0].get()
            u1 = self._vars["u1"][0].get()
            u2 = self._vars["u2"][0].get()
            e  = self._vars["e"][0].get()
            v1 = ((m1 - e * m2) * u1 + (1 + e) * m2 * u2) / (m1 + m2)
            v2 = ((m2 - e * m1) * u2 + (1 + e) * m1 * u1) / (m1 + m2)
            Ti = 0.5 * m1 * u1**2 + 0.5 * m2 * u2**2
            Tf = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
            dT = Tf - Ti
            if np.isclose(Ti, Tf, atol=1e-5):    ctype = "Elastic"
            elif e == 0:                           ctype = "Perf. Inelastic"
            else:                                  ctype = "Inelastic"
            self._lbl_v1   .config(text=f"{v1:.2f} m/s")
            self._lbl_v2   .config(text=f"{v2:.2f} m/s")
            self._lbl_dT   .config(text=f"{dT:.2f} J")
            self._lbl_ctype.config(text=ctype)
        except Exception:
            pass

    def _update_results_box(self, s):
        lines = [
            f"{'─'*26}",
            f" u₁ = {s['u1']:>7.3f}  →  v₁ = {s['v1']:>7.3f} m/s",
            f" u₂ = {s['u2']:>7.3f}  →  v₂ = {s['v2']:>7.3f} m/s",
            f"{'─'*26}",
            f" KE initial  = {s['Ti']:>9.3f} J",
            f"   KE₁ = {s['Ti1']:>9.3f} J",
            f"   KE₂ = {s['Ti2']:>9.3f} J",
            f" KE final    = {s['Tf']:>9.3f} J",
            f"   KE₁ = {s['Tf1']:>9.3f} J",
            f"   KE₂ = {s['Tf2']:>9.3f} J",
            f" ΔKE         = {s['delta_T']:>+9.3f} J",
            f"{'─'*26}",
            f" p initial   = {s['Pi']:>9.3f} kg·m/s",
            f" p final     = {s['Pf']:>9.3f} kg·m/s",
            f" Δp          = {s['delta_P']:>+9.5f}",
            f"{'─'*26}",
            f" Type: {s['ctype']}",
        ]
        self._results_box.config(state="normal")
        self._results_box.delete("1.0", "end")
        self._results_box.insert("end", "\n".join(lines))
        self._results_box.config(state="disabled")

#Tabs
    def _set_tab(self, key):
        self._cur_tab = key
        self._refresh_tabs()
        if self._sim:
            self._stop()
            self._draw_full()

    def _refresh_tabs(self):
        for k, btn in self._tab_btns.items():
            active = k == self._cur_tab
            btn.config(bg=TEXT_PRI if active else BG,
                       fg="white"  if active else TEXT_SEC,
                       highlightthickness=1,
                       highlightbackground=TEXT_PRI if active else BORDER)

    # Run/Stop
    def _run(self):
        self._stop()
        p = {k: v.get() for k, (v, _) in self._vars.items()}
        self._sim = build_sim(p["m1"], p["m2"], p["u1"], p["u2"], p["e"])
        self._update_stats()
        self._update_results_box(self._sim)
        self._badge.config(text="running", bg=BADGE_RUN, fg=BADGE_RUN_FG)

        if self._cur_tab == "anim":
            self._run_animation()
        else:
            self._draw_full()
            self._badge.config(text="done", bg=CARD_BG, fg=BADGE_IDL_FG)

    def _run_animation(self):
        s     = self._sim
        total = len(s["times"])
        step  = max(1, total // 400)

        self._clear_fig()
        ax_top = self._fig.add_subplot(2, 1, 1)
        ax_bot = self._fig.add_subplot(2, 1, 2)
        self._fig.tight_layout(pad=1.8, h_pad=2.5)

        ax_top.set_facecolor(BG)
        for sp in ax_top.spines.values():
            sp.set_color(BORDER)
        ax_top.set_xlim(-20, 20)
        ax_top.set_ylim(-2, 2)
        ax_top.set_yticks([])
        ax_top.set_xlabel("Position (m)", color=TEXT_SEC, fontsize=8)
        ax_top.tick_params(colors=TEXT_SEC, labelsize=8)
        ax_top.axhline(0, color=BORDER, lw=1)
        ax_top.set_title("1D Collision Animation", color=TEXT_PRI,
                          fontsize=10, pad=6)

        blk1 = Rectangle((s["pos1"][0], -0.5), BLOCK_SIZE, 1,
                          facecolor=ACCENT, edgecolor="white", lw=1.2, zorder=3)
        blk2 = Rectangle((s["pos2"][0], -0.5), BLOCK_SIZE, 1,
                          facecolor=RED_ACC, edgecolor="white", lw=1.2, zorder=3)
        ax_top.add_patch(blk1)
        ax_top.add_patch(blk2)
        lbl1 = ax_top.text(s["pos1"][0] + BLOCK_SIZE / 2, 0,
                           "m₁", ha="center", va="center",
                           color="white", fontsize=8, fontweight="bold", zorder=4)
        lbl2 = ax_top.text(s["pos2"][0] + BLOCK_SIZE / 2, 0,
                           "m₂", ha="center", va="center",
                           color="white", fontsize=8, fontweight="bold", zorder=4)

        vel_txt = ax_top.text(0.02, 0.88, "",
                              transform=ax_top.transAxes,
                              color=TEXT_PRI, fontsize=9,
                              fontfamily="monospace")

        # collision marker line (hidden until impact)
        coll_line = ax_top.axvline(x=0, color=ORANGE, lw=1,
                                   ls="--", alpha=0, zorder=2)
        
        self._style_ax(ax_bot)
        ax_bot.set_xlim(0, T_MAX)
        all_pos = np.concatenate([s["pos1"], s["pos2"]])
        ax_bot.set_ylim(np.min(all_pos) - 2, np.max(all_pos) + 2)
        ax_bot.set_xlabel("Time (s)", color=TEXT_SEC, fontsize=8)
        ax_bot.set_ylabel("Position (m)", color=TEXT_SEC, fontsize=8)
        ax_bot.set_title("Displacement vs Time", color=TEXT_PRI,
                          fontsize=10, pad=6)
        line1, = ax_bot.plot([], [], color=ACCENT,  lw=1.6, label="Body 1")
        line2, = ax_bot.plot([], [], color=RED_ACC, lw=1.6, label="Body 2")
        ax_bot.legend(fontsize=8, facecolor=PANEL_BG,
                      edgecolor=BORDER, labelcolor=TEXT_PRI)
        cf = s["collision_frame"]

        def update(i):
            n = min((i + 1) * step, total)
            f = n - 1

            blk1.set_x(s["pos1"][f])
            blk2.set_x(s["pos2"][f])
            lbl1.set_x(s["pos1"][f] + BLOCK_SIZE / 2)
            lbl2.set_x(s["pos2"][f] + BLOCK_SIZE / 2)

            if cf is not None and f >= cf:
                cv1, cv2 = s["v1"], s["v2"]
                coll_line.set_xdata([s["pos1"][cf] + BLOCK_SIZE])
                coll_line.set_alpha(0.6)
            else:
                cv1, cv2 = s["u1"], s["u2"]

            vel_txt.set_text(f"v₁ = {cv1:+.2f} m/s    v₂ = {cv2:+.2f} m/s")

            line1.set_data(s["times"][:n], s["pos1"][:n])
            line2.set_data(s["times"][:n], s["pos2"][:n])

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

    # Static chart tabs
    def _draw_full(self):
        s = self._sim
        self._clear_fig()
        ax = self._fig.add_subplot(111)
        self._style_ax(ax)
        self._fig.tight_layout(pad=1.8)

        t = s["times"]

        if self._cur_tab == "e":
            ax.plot(t, s["ke1"],     color=ACCENT,  lw=1.5, label="KE₁")
            ax.plot(t, s["ke2"],     color=RED_ACC, lw=1.5, label="KE₂")
            ax.plot(t, s["ke_total"], color=GREEN,  lw=1.8, ls="--", label="KE total")
            if s["collision_frame"]:
                ct = s["times"][s["collision_frame"]]
                ax.axvline(ct, color=ORANGE, lw=1, ls=":", alpha=0.8,
                           label=f"collision @ {ct:.2f}s")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Kinetic Energy (J)")
            ax.set_title("Kinetic Energy vs Time", color=TEXT_PRI, fontsize=10)
            ax.set_xlim(0, T_MAX)
            ax.legend(fontsize=8, facecolor=PANEL_BG,
                      edgecolor=BORDER, labelcolor=TEXT_PRI)
        elif self._cur_tab == "p":
            ax.plot(t, s["p1"],      color=ACCENT,  lw=1.5, label="p₁")
            ax.plot(t, s["p2"],      color=RED_ACC, lw=1.5, label="p₂")
            ax.plot(t, s["p_total"], color=GREEN,   lw=1.8, ls="--", label="p total")
            if s["collision_frame"]:
                ct = s["times"][s["collision_frame"]]
                ax.axvline(ct, color=ORANGE, lw=1, ls=":", alpha=0.8,
                           label=f"collision @ {ct:.2f}s")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Momentum (kg·m/s)")
            ax.set_title("Momentum vs Time", color=TEXT_PRI, fontsize=10)
            ax.set_xlim(0, T_MAX)
            ax.legend(fontsize=8, facecolor=PANEL_BG,
                      edgecolor=BORDER, labelcolor=TEXT_PRI)

        elif self._cur_tab == "disp":
            ax.plot(t, s["pos1"], color=ACCENT,  lw=1.6, label="Body 1")
            ax.plot(t, s["pos2"], color=RED_ACC, lw=1.6, label="Body 2")
            ax.fill_between(t, s["pos1"], s["pos2"],
                            color=PURPLE, alpha=0.06)
            if s["collision_frame"]:
                ct = s["times"][s["collision_frame"]]
                ax.axvline(ct, color=ORANGE, lw=1, ls=":", alpha=0.8,
                           label=f"collision @ {ct:.2f}s")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
            ax.set_title("Displacement vs Time", color=TEXT_PRI, fontsize=10)
            ax.set_xlim(0, T_MAX)
            ax.legend(fontsize=8, facecolor=PANEL_BG,
                      edgecolor=BORDER, labelcolor=TEXT_PRI)
        self._mpl_canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app  = CollisionApp(root)
    root.mainloop()
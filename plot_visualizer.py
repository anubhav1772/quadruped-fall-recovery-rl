import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import json, sys, os, re
from pathlib import Path
import matplotlib
import plotly.graph_objs as go

# ------------------------------------------------------------------
# GLOBAL MATPLOTLIB STYLE
# ------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 4),
    "lines.linewidth": 2.5,
})
plt.style.use("seaborn-v0_8-darkgrid")


# ---------------------------------------------------------
# Parse log file
# ---------------------------------------------------------
def parse_log_file(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    data = []
    for l in lines:
        m = re.match(r".*│\s*(.*?)\s*│\s*([-\d\.]+)\s*│", l)
        if m:
            name, val = m.groups()
            try:
                data.append((name, float(val)))
            except:
                pass

    if not data:
        raise RuntimeError("No metrics found")

    df = pd.DataFrame(data, columns=["metric", "value"])

    iters = []
    cur = 0
    for name, val in data:
        if name.strip() == "iterations":
            cur = int(val)
        iters.append(cur)

    df["iter"] = iters
    df = df[df.metric != "iterations"]

    return df


# ---------------------------------------------------------
# App
# ---------------------------------------------------------
class MetricsApp:
    HISTORY_LIMIT = 8

    def __init__(self, root):
        self.root = root
        self.root.title("Interactive RL Metrics Viewer")
        self.root.geometry("1300x850")

        self.df = None
        self.metrics = []
        self.df_file = None
        self.dark = True  # start in dark mode

        self.auto_refresh = tk.BooleanVar(value=False)
        self.auto_refresh_interval = tk.IntVar(value=10)
        self._refresh_job = None

        # ttk Style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.load_history()

        self.build_menu()
        self.build_layout()

        self.apply_ui_theme()
        self.update_recent_dropdown()
        self.bind_shortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.stat_cache = {}

    # -----------------------------------------------------
    # UI theme
    # -----------------------------------------------------
    def apply_ui_theme(self):
        if self.dark:
            bg = "#202225"
            fg = "#ffffff"
            accent = "#2f3136"
            btn_active = "#3c3f41"
        else:
            bg = "#f0f0f0"
            fg = "#000000"
            accent = "#e0e0e0"
            btn_active = "#d0d0d0"

        self.root.configure(bg=bg)

        base_font = ("Segoe UI", 12)
        self.style.configure(".", font=base_font)

        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg, font=base_font)
        self.style.configure("TButton", background=accent, foreground=fg, font=base_font)
        self.style.map("TButton",
                       background=[("active", btn_active)],
                       foreground=[("active", fg)])
        self.style.configure("TNotebook.Tab", background=accent, foreground=fg, font=base_font)

        self.style.configure("Treeview",
                             background=accent,
                             foreground=fg,
                             fieldbackground=accent,
                             rowheight=24,
                             font=base_font)

        self.style.configure("Treeview.Heading",
                             background=accent,
                             foreground=fg,
                             font=("Segoe UI", 13, "bold"))

        if hasattr(self, "stats_text"):
            self.stats_text.configure(bg=accent, fg=fg, insertbackground=fg)
        if hasattr(self, "raw_text"):
            self.raw_text.configure(bg=accent, fg=fg, insertbackground=fg)

    # -----------------------------------------------------
    # Persistent history
    # -----------------------------------------------------
    def load_history(self):
        self.history_file = Path.home() / ".rl_metrics_viewer_history.json"
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except:
                self.history = []
        else:
            self.history = []

    def save_history(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f)
        except:
            pass

    def add_to_history(self, path):
        path = str(Path(path).resolve())
        if path in self.history:
            self.history.remove(path)
        self.history.insert(0, path)
        self.history = self.history[:self.HISTORY_LIMIT]
        self.save_history()
        self.update_recent_menu()
        self.update_recent_dropdown()

    def clear_history(self):
        self.history = []
        self.save_history()
        self.update_recent_menu()
        self.update_recent_dropdown()

    # -----------------------------------------------------
    # Menu Bar
    # -----------------------------------------------------
    def build_menu(self):
        self.menubar = tk.Menu(self.root)

        f = tk.Menu(self.menubar, tearoff=0)
        f.add_command(label="Open", command=self.browse, accelerator="Ctrl+O")

        self.recent_menu = tk.Menu(f, tearoff=0)
        f.add_cascade(label="Open Recent", menu=self.recent_menu)

        f.add_command(label="Reload", command=self.reload_file, accelerator="Ctrl+R")
        f.add_command(label="Export Plotly HTML", command=self.export_plotly_html)
        f.add_separator()
        f.add_command(label="Exit", command=self.on_close, accelerator="Ctrl+Q")
        self.menubar.add_cascade(label="File", menu=f)

        v = tk.Menu(self.menubar, tearoff=0)
        v.add_command(label="Toggle Dark Mode", command=self.toggle_dark)
        v.add_command(label="Compare Runs (Plotly)", command=self.compare_runs_plotly)
        self.menubar.add_cascade(label="View", menu=v)

        self.root.config(menu=self.menubar)
        self.update_recent_menu()

    def update_recent_menu(self):
        self.recent_menu.delete(0, "end")

        if not self.history:
            self.recent_menu.add_command(label="(empty)", state="disabled")
            return

        for path in self.history:
            name = os.path.basename(path)
            self.recent_menu.add_command(
                label=f"{name}   ({path})",
                command=lambda p=path: self.open_recent(p)
            )

        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Clear History", command=self.clear_history)

    def open_recent(self, path):
        self.path_var.set(path)
        self.load_file()

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    def build_layout(self):
        # Top bar
        top = ttk.Frame(self.root, padding=5)
        top.pack(fill="x")

        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=70).pack(side="left", padx=5)

        ttk.Button(top, text="Browse", command=self.browse).pack(side="left", padx=5)
        ttk.Button(top, text="Load", command=self.load_file).pack(side="left", padx=5)

        # -----------------------------------------------
        # ADD DARK/LIGHT TOGGLE BUTTON
        # -----------------------------------------------
        ttk.Button(top, text="Dark / Light", command=self.toggle_dark).pack(
            side="left", padx=10
        )
        # -----------------------------------------------

        # Recent dropdown
        self.recent_var = tk.StringVar(value="Open Recent")
        self.recent_combo = ttk.Combobox(top, textvariable=self.recent_var, state="readonly", width=40)
        self.recent_combo.pack(side="left", padx=5)
        self.recent_combo.bind("<<ComboboxSelected>>", self.choose_recent)

        # Tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.tab_plot = ttk.Frame(self.tabs)
        self.tab_stats = ttk.Frame(self.tabs)
        self.tab_raw = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_plot, text="Plots")
        self.tabs.add(self.tab_stats, text="Stats")
        self.tabs.add(self.tab_raw, text="Raw Data")

        # Split plot tab left/right
        paned = ttk.Panedwindow(self.tab_plot, orient="horizontal")
        paned.pack(fill="both", expand=True)

        side = ttk.Frame(paned, width=350, padding=5)
        paned.add(side, weight=1)

        area = ttk.Frame(paned, padding=5)
        paned.add(area, weight=5)

        # Sidebar label + filter
        ttk.Label(side, text="Metrics:").pack(anchor="w")

        filter_frame = ttk.Frame(side)
        filter_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(filter_frame, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=20)
        filter_entry.pack(side="left", fill="x", expand=True)
        filter_entry.bind("<KeyRelease>", lambda e: self.update_metric_list())

        # Sidebar treeview
        tree_frame = ttk.Frame(side)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("metric",), show="headings", selectmode="extended")
        self.tree.heading("metric", text="Metric Name")
        self.tree.column("metric", width=330, anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # Sorting on header click
        def sort_tree(col, reverse=False):
            data = [(self.tree.item(i,"values")[0], i) for i in self.tree.get_children("")]
            data.sort(reverse=reverse)
            for idx, (_, item) in enumerate(data):
                self.tree.move(item, "", idx)
            self.tree.heading(col, command=lambda: sort_tree(col, not reverse))

        self.tree.heading("metric", command=lambda: sort_tree("metric", False))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.update_plot())

        # Plot area + toolbar
        self.fig, self.ax = plt.subplots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.annot = None
        self.vline = None
        self.hline = None
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        self.toolbar = NavigationToolbar2Tk(self.canvas, area)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        # Controls under sidebar
        ctl = ttk.Frame(side, padding=5)
        ctl.pack(fill="x", pady=6)

        ttk.Label(ctl, text="Plot style:").pack(anchor="w")
        self.style_var = tk.StringVar(value="line")
        ttk.Radiobutton(
            ctl, text="Line", value="line", variable=self.style_var, command=self.update_plot
        ).pack(anchor="w")
        ttk.Radiobutton(
            ctl, text="Dots", value="dots", variable=self.style_var, command=self.update_plot
        ).pack(anchor="w")

        ttk.Label(ctl, text="Moving Avg Window:").pack(anchor="w")
        self.smooth = tk.DoubleVar(value=1.0)
        tk.Scale(
            ctl, from_=1, to=200, orient="horizontal", variable=self.smooth,
            command=lambda x: self.update_plot()
        ).pack(fill="x")

        ttk.Button(ctl, text="Refresh (Reload File)", command=self.reload_file).pack(
            pady=4, fill="x"
        )
        ttk.Button(ctl, text="Save Current Plot", command=self.save_plot).pack(
            pady=4, fill="x"
        )
        ttk.Button(ctl, text="Open Plotly Dashboard", command=self.open_plotly_dashboard).pack(
            pady=4, fill="x"
        )

        # Auto-refresh controls
        ttk.Checkbutton(
            ctl,
            text="Auto-refresh",
            variable=self.auto_refresh,
            command=self.schedule_auto_refresh
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(ctl, text="Interval (s):").pack(anchor="w")
        self.interval_spin = ttk.Spinbox(
            ctl,
            from_=2,
            to=3600,
            textvariable=self.auto_refresh_interval,
            width=6
        )
        self.interval_spin.pack(anchor="w")

        # Stats and raw data tabs
        self.stats_text = tk.Text(self.tab_stats, font=("consolas", 14))
        self.stats_text.pack(fill="both", expand=True)

        self.raw_text = tk.Text(self.tab_raw, font=("consolas", 12))
        self.raw_text.pack(fill="both", expand=True)

        # Status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")

        self.cursor_label = tk.StringVar(value="")
        self.point_label = tk.StringVar(value="")

        hud_frame = ttk.Frame(self.root)
        hud_frame.pack(anchor="se", fill="x")

        ttk.Label(hud_frame, textvariable=self.cursor_label, anchor="e").pack(side="right", padx=10)
        ttk.Label(hud_frame, textvariable=self.point_label, anchor="e").pack(side="right", padx=10)

        ttk.Label(ctl, text="X-Axis Mode:").pack(anchor="w")

        self.xmode = tk.StringVar(value="iteration")

        ttk.Radiobutton(
            ctl, text="Iteration",
            variable=self.xmode, value="iteration",
            command=self.update_plot
        ).pack(anchor="w")

        ttk.Radiobutton(
            ctl, text="Velocity (CoT vs v)",
            variable=self.xmode, value="velocity",
            command=self.update_plot
        ).pack(anchor="w")


    def get_cot_vs_velocity(self):
        """
        Returns arrays:
        vx_cmd_abs, cot, tracking_error (optional)
        """
        df = self.df

        # required metrics (adjust names if needed)
        vx_cmd = df[df.metric == "train/episode/vx cmd abs/mean"]
        vx     = df[df.metric == "train/episode/vx abs/mean"]
        cot    = df[df.metric == "train/episode/CoT(Unitless)/mean"]

        if vx_cmd.empty or vx.empty or cot.empty:
            return None

        # align by iteration
        merged = (
            vx_cmd.merge(vx, on="iter", suffixes=("_cmd", "_act"))
                  .merge(cot, on="iter")
        )

        vx_cmd_abs = merged["value_cmd"].to_numpy()
        cot_val    = merged["value"].to_numpy()

        tracking_err = np.abs(
            merged["value_cmd"] - merged["value_act"]
        ) / np.maximum(merged["value_cmd"], 1e-3)

        return vx_cmd_abs, cot_val, tracking_err



    # -----------------------------------------------------
    # Recent dropdown
    # -----------------------------------------------------
    def update_recent_dropdown(self):
        if not self.history:
            self.recent_combo["values"] = []
            self.recent_var.set("Open Recent")
            return

        display = [os.path.basename(p) for p in self.history]
        self.recent_combo["values"] = display
        self.recent_var.set("Open Recent")

    def choose_recent(self, event):
        idx = self.recent_combo.current()
        if idx < 0 or idx >= len(self.history):
            return
        path = self.history[idx]
        self.path_var.set(path)
        self.load_file()

    # -----------------------------------------------------
    # File ops
    # -----------------------------------------------------
    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Log", "*.log"), ("All", "*.*")]
        )
        if path:
            self.path_var.set(path)

    def load_file(self):
        path = self.path_var.get()
        if not path:
            return

        try:
            self.df = parse_log_file(path)
            self.df_file = path
            self.metrics = sorted(self.df.metric.unique())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.add_to_history(path)

        self.update_metric_list()
        self.update_stats()
        self.update_raw()
        self.update_plot()

        self.status.set(f"Loaded {len(self.metrics)} metrics from {os.path.basename(path)}.")

        self.stat_cache.clear()

    def reload_file(self):
        if self.df_file:
            self.load_file()

    # -----------------------------------------------------
    # Sidebar population
    # -----------------------------------------------------
    def update_metric_list(self):
        self.tree.delete(*self.tree.get_children())
        if not self.metrics:
            return

        filter_text = self.filter_var.get().strip().lower()
        for m in self.metrics:
            if filter_text and filter_text not in m.lower():
                continue
            self.tree.insert("", "end", values=(m,))

    # -----------------------------------------------------
    # Stats & Raw
    # -----------------------------------------------------
    def update_stats(self):
        if self.df is None:
            return

        g = self.df.groupby("metric")["value"].agg(["mean", "std", "min", "max"])
        text = "Metric Statistics:\n\n" + str(g.round(3))

        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", text)

    def update_raw(self):
        if self.df is None:
            return

        text = self.df.head(200).to_string()
        self.raw_text.delete("1.0", "end")
        self.raw_text.insert("end", text)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    # def update_plot(self):
        # if self.df is None:
        #     return

        # selected_ids = self.tree.selection()
        # if not selected_ids:
        #     return

        # selected = [self.tree.item(i, "values")[0] for i in selected_ids]

        # self.fig.clear()

        # n = len(selected)

        # if n == 1:
        #     axes = [self.fig.add_subplot(111)]
        # else:
        #     axes = self.fig.subplots(nrows=n, ncols=1, sharex=True)
        #     if not isinstance(axes, (list, np.ndarray)):
        #         axes = [axes]

        # window = max(1, int(self.smooth.get()))

    #     for ax, metric in zip(axes, selected):
    #         sub = self.df[self.df.metric == metric]
    #         x = sub.iter.to_numpy()
    #         y = sub.value.to_numpy()

    #         s = pd.Series(y)
    #         mean = s.rolling(window, min_periods=1).mean()
    #         std = s.rolling(window, min_periods=1).std().fillna(0)

    #         if self.style_var.get() == "line":
    #             ax.plot(x, mean, label=f"{metric} (MA)")
    #         else:
    #             ax.scatter(x, mean, s=10, label=f"{metric} (MA)")

    #         ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    #         ax.set_ylabel(metric)
    #         ax.grid(alpha=0.3)
    #         ax.legend()

    #     axes[-1].set_xlabel("Iteration")

    #     self.apply_theme_to_axes()
    #     self.fig.tight_layout()

    #     # reset crosshair objects + tooltip + labels
    #     self.vline = None
    #     self.hline = None
    #     self.annot = None
    #     self.cursor_label.set("")
    #     self.point_label.set("")

    #     self.canvas.draw()

    #     self.status.set(f"Plotted {len(selected)} metrics (window={window}).")

    def get_metric_stats(self, metric, window):
        key = (metric, window)
        if key in self.stat_cache:
            return self.stat_cache[key]

        # gait metric
        if metric.endswith(" mean/mean"):
            std_metric = metric.replace(" mean/mean", " std/mean")

            m = self.df[self.df.metric == metric]
            s = self.df[self.df.metric == std_metric]

            if m.empty or s.empty:
                return None

            x = m.iter.to_numpy()
            mean = m.value.to_numpy()
            std = s.value.to_numpy()

        # normal metric
        else:
            sub = self.df[self.df.metric == metric]
            x = sub.iter.to_numpy()
            y = sub.value.to_numpy()

            series = pd.Series(y)
            mean = series.rolling(window, min_periods=1).mean().to_numpy()
            std = series.rolling(window, min_periods=1).std().fillna(0).to_numpy()

        lower = mean - std
        upper = mean + std

        self.stat_cache[key] = (x, mean, lower, upper)
        return x, mean, lower, upper


    def update_plot(self):
        """
        Extra: plot gait params with std bands.
        """
        if self.xmode.get() == "velocity":
            self.plot_cot_vs_velocity()
            return

        if self.df is None:
            return

        selected_ids = self.tree.selection()
        if not selected_ids:
            return

        selected = [self.tree.item(i, "values")[0] for i in selected_ids]

        self.fig.clear()

        n = len(selected)
        if n == 1:
            axes = [self.fig.add_subplot(111)]
        else:
            axes = self.fig.subplots(nrows=n, ncols=1, sharex=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

        window = max(1, int(self.smooth.get()))

        for ax, metric in zip(axes, selected):

            stats = self.get_metric_stats(metric, window)
            if stats is None:
                continue

            x, mean, lower, upper = stats

            # Optional decimation for speed
            STEP = max(1, len(x) // 2000)
            x = x[::STEP]
            mean = mean[::STEP]
            lower = lower[::STEP]
            upper = upper[::STEP]

            ax.plot(x, mean, label=metric)

            # Show band only if meaningful
            # if window >= 2:
            ax.fill_between(x, lower, upper, alpha=0.25)

            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Iteration")
        self.apply_theme_to_axes()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_cot_vs_velocity(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        data = self.get_cot_vs_velocity()
        if data is None:
            ax.text(0.5, 0.5, "Required metrics not found",
                    ha="center", va="center")
            self.canvas.draw()
            return

        vx, cot, terr = data

        # optional binning
        bins = np.linspace(vx.min(), vx.max(), 10)
        bin_ids = np.digitize(vx, bins)

        vx_b, cot_b = [], []
        for b in range(1, len(bins)):
            mask = bin_ids == b
            if np.any(mask):
                vx_b.append(vx[mask].mean())
                cot_b.append(cot[mask].mean())

        sc = ax.scatter(
            vx, cot,
            c=terr,
            cmap="viridis",
            alpha=0.5,
            label="Episodes"
        )

        ax.plot(vx_b, cot_b, "-o", color="white", label="Binned mean")

        ax.set_xlabel("Commanded velocity |vx| (m/s)")
        ax.set_ylabel("Cost of Transport (CoT)")
        ax.set_yscale("log")   # VERY IMPORTANT for CoT analysis
        ax.grid(alpha=0.3)
        ax.legend()

        cbar = self.fig.colorbar(sc, ax=ax)
        cbar.set_label("Tracking error")

        self.apply_theme_to_axes()
        self.fig.tight_layout()
        self.canvas.draw_idle()


    # -----------------------------------------------------
    # Save figure (matplotlib)
    # -----------------------------------------------------
    def save_plot(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.fig.savefig(file_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save figure: {e}")
            return

        self.status.set(f"Saved plot to {file_path}.")

    # -----------------------------------------------------
    # Plotly helpers
    # -----------------------------------------------------
    def create_plotly_figure(self, selected_metrics=None):
        if self.df is None:
            return None

        if selected_metrics is None or not selected_metrics:
            selected_metrics = self.metrics[:10]

        fig = go.Figure()
        for metric in selected_metrics:
            sub = self.df[self.df.metric == metric]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["iter"],
                y=sub["value"],
                mode="lines",
                name=metric
            ))

        fig.update_layout(
            title="RL Metrics",
            xaxis_title="Iteration",
            yaxis_title="Value",
            template="plotly_dark" if self.dark else "plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def open_plotly_dashboard(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return

        selected_ids = self.tree.selection()
        if selected_ids:
            selected = [self.tree.item(i, "values")[0] for i in selected_ids]
        else:
            selected = self.metrics[:10]

        fig = self.create_plotly_figure(selected)
        if fig is None:
            messagebox.showwarning("Warning", "No data for selected metrics.")
            return

        base_dir = os.path.dirname(self.df_file) if self.df_file else os.getcwd()
        tmp_path = os.path.join(base_dir, "metrics_interactive.html")
        try:
            fig.write_html(tmp_path, auto_open=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Plotly dashboard: {e}")
            return

        self.status.set(f"Opened interactive dashboard: {tmp_path}")

    def export_plotly_html(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data to export.")
            return

        selected_ids = self.tree.selection()
        if selected_ids:
            selected = [self.tree.item(i, "values")[0] for i in selected_ids]
        else:
            selected = self.metrics

        fig = self.create_plotly_figure(selected)
        if fig is None:
            messagebox.showwarning("Warning", "No data for selected metrics.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML file", "*.html"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            fig.write_html(file_path, auto_open=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export HTML: {e}")
            return

        self.status.set(f"Exported interactive dashboard to {file_path}.")

    def compare_runs_plotly(self):
        # Select metric to compare
        selected_ids = self.tree.selection()
        if not selected_ids:
            messagebox.showinfo("Info", "Select a metric in the list to compare across runs.")
            return
        metric = self.tree.item(selected_ids[0], "values")[0]

        # Ask for additional log files
        paths = filedialog.askopenfilenames(
            title="Select log files to compare",
            filetypes=[("Log", "*.log"), ("All files", "*.*")]
        )
        if not paths:
            return

        runs = []
        if self.df is not None and self.df_file:
            runs.append((os.path.basename(self.df_file), self.df))

        for p in paths:
            try:
                df_other = parse_log_file(p)
                runs.append((os.path.basename(p), df_other))
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to parse {p}: {e}")

        if not runs:
            messagebox.showwarning("Warning", "No valid runs to compare.")
            return

        fig = go.Figure()
        for name, df_run in runs:
            sub = df_run[df_run.metric == metric]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["iter"],
                y=sub["value"],
                mode="lines",
                name=name
            ))

        if not fig.data:
            messagebox.showwarning("Warning", f"No data for metric '{metric}' in selected runs.")
            return

        fig.update_layout(
            title=f"Compare '{metric}' across runs",
            xaxis_title="Iteration",
            yaxis_title=metric,
            template="plotly_dark" if self.dark else "plotly_white"
        )

        tmp_path = os.path.join(os.getcwd(), f"compare_{metric}.html")
        try:
            fig.write_html(tmp_path, auto_open=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open comparison dashboard: {e}")
            return

        self.status.set(f"Opened comparison dashboard: {tmp_path}")

    # -----------------------------------------------------
    # Theme for Matplotlib axes
    # -----------------------------------------------------
    def apply_theme_to_axes(self):

        if self.dark:
            self.fig.patch.set_facecolor("#222222")
            for ax in self.fig.axes:
                ax.set_facecolor("#222222")
                ax.tick_params(colors="white")
                ax.yaxis.label.set_color("white")
                ax.xaxis.label.set_color("white")
                ax.title.set_color("white")

                for spine in ax.spines.values():
                    spine.set_edgecolor("white")

                leg = ax.get_legend()
                if leg:
                    leg.get_frame().set_facecolor("#222222")
                    leg.get_frame().set_edgecolor("white")
                    for text in leg.get_texts():
                        text.set_color("white")

        else:
            self.fig.patch.set_facecolor("white")
            for ax in self.fig.axes:
                ax.set_facecolor("white")
                ax.tick_params(colors="black")
                ax.yaxis.label.set_color("black")
                ax.xaxis.label.set_color("black")
                ax.title.set_color("black")

                for spine in ax.spines.values():
                    spine.set_edgecolor("black")

                leg = ax.get_legend()
                if leg:
                    leg.get_frame().set_facecolor("white")
                    leg.get_frame().set_edgecolor("black")
                    for text in leg.get_texts():
                        text.set_color("black")

    # -----------------------------------------------------
    # Toggle Dark/Light
    # -----------------------------------------------------
    def toggle_dark(self):
        self.dark = not self.dark
        self.apply_ui_theme()
        self.apply_theme_to_axes()
        self.canvas.draw()

    def on_hover(self, event):
        """Crosshair + tooltip on nearest data point (pixel-based distance)."""

        # No data / mouse not over an axes
        if self.df is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes

        closest = None
        min_dist = float("inf")

        # Mouse position in display coords (pixels)
        mx, my = ax.transData.transform((event.xdata, event.ydata))

        # --- find closest data point across all Line2D objects (skip crosshair lines) ---
        for line in ax.get_lines():
            # Skip our own crosshair lines if they exist
            if line is self.vline or line is self.hline:
                continue

            xdata = np.array(line.get_xdata(orig=False))
            ydata = np.array(line.get_ydata(orig=False))

            if xdata.size == 0:
                continue

            # nearest X index
            idx = np.searchsorted(xdata, event.xdata)
            idx = np.clip(idx, 0, xdata.size - 1)

            x, y = xdata[idx], ydata[idx]

            # data point in display coords
            px, py = ax.transData.transform((x, y))

            dist = np.hypot(px - mx, py - my)  # pixel distance

            if dist < min_dist:
                min_dist = dist
                closest = (x, y)

        if closest is None:
            return

        x, y = closest

        # ---- threshold in PIXELS: if too far from any line, hide HUD ---
        PIX_THRESHOLD = 20  # ~20 px
        if min_dist > PIX_THRESHOLD:
            if self.annot:
                self.annot.set_visible(False)
            if self.vline:
                self.vline.set_visible(False)
            if self.hline:
                self.hline.set_visible(False)

            self.canvas.draw_idle()
            # still update mouse label, but no nearest point
            self.cursor_label.set(f"mouse: x={event.xdata:.3f}, y={event.ydata:.3f}")
            self.point_label.set("")
            return

        # -----------------------------------------------------
        # --- CREATE/UPDATE bbox tooltip near the point ---
        # -----------------------------------------------------
        if self.annot is None:
            fc = "#333333" if self.dark else "white"
            ec = "white" if self.dark else "black"
            tc = "white" if self.dark else "black"

            self.annot = ax.annotate(
                "",
                xy=(x, y),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, alpha=0.9),
                color=tc
            )

        self.annot.xy = (x, y)
        self.annot.set_text(f"x = {x:.3f}\ny = {y:.3f}")
        self.annot.set_visible(True)

        # -----------------------------------------------------
        # --- CREATE / UPDATE CROSSHAIR LINES ---
        # -----------------------------------------------------
        clr = "white" if self.dark else "gray"

        if self.vline is None:
            self.vline = ax.axvline(x, linestyle="--", linewidth=1, color=clr, zorder=1000)
        if self.hline is None:
            self.hline = ax.axhline(y, linestyle="--", linewidth=1, color=clr, zorder=1000)

        # move lines
        self.vline.set_xdata([x, x])
        self.hline.set_ydata([y, y])
        self.vline.set_visible(True)
        self.hline.set_visible(True)

        # redraw
        self.canvas.draw_idle()

        # -----------------------------------------------------
        # bottom HUD: show both raw mouse + nearest data point
        # -----------------------------------------------------
        self.cursor_label.set(f"mouse: x={event.xdata:.3f}, y={event.ydata:.3f}")
        self.point_label.set(f"nearest: x={x:.3f}, y={y:.3f}")


    # -----------------------------------------------------
    # Auto-refresh
    # -----------------------------------------------------
    def schedule_auto_refresh(self):
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except Exception:
                pass
            self._refresh_job = None

        if self.auto_refresh.get():
            self._schedule_next_refresh()

    def _schedule_next_refresh(self):
        if not self.auto_refresh.get():
            return
        try:
            interval_sec = max(2, int(self.auto_refresh_interval.get()))
        except Exception:
            interval_sec = 10
            self.auto_refresh_interval.set(interval_sec)
        self._refresh_job = self.root.after(interval_sec * 1000, self._auto_refresh_callback)

    def _auto_refresh_callback(self):
        if self.df_file:
            try:
                self.load_file()
                self.status.set(f"Auto-refreshed: {os.path.basename(self.df_file)}")
            except Exception as e:
                self.status.set(f"Auto-refresh failed: {e}")
        self._schedule_next_refresh()

    # -----------------------------------------------------
    # Keyboard shortcuts
    # -----------------------------------------------------
    def bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.browse())
        self.root.bind("<Control-r>", lambda e: self.reload_file())
        self.root.bind("<Control-q>", lambda e: self.on_close())
        self.root.bind("<Control-d>", lambda e: self.toggle_dark())

    # -----------------------------------------------------
    # Exit
    # -----------------------------------------------------
    def on_close(self):
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except Exception:
                pass
            self._refresh_job = None
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetricsApp(root)
    root.mainloop()

#!/usr/bin/env python3
"""
plot_pi.py  -  Plot Perfusion Index from dual Innovo IP900BPB overnight recording.
Usage:  python plot_pi.py [path/to/summary.csv]
        (defaults to 20260218_223711_summary.csv in the same directory)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector

# -- Load data ----------------------------------------------------------------
csv_path = sys.argv[1] if len(sys.argv) > 1 else "20260218_223711_summary.csv"
if not os.path.exists(csv_path):
    sys.exit(f"File not found: {csv_path}")

df = pd.read_csv(csv_path, parse_dates=["timestamp"])
df = df.dropna(subset=["pi_unit1", "pi_unit2"])
df = df.sort_values("timestamp").reset_index(drop=True)

t  = df["timestamp"]
p1 = df["pi_unit1"]
p2 = df["pi_unit2"]

C1, C2 = "#1f77b4", "#d62728"   # blue, red

# -- Figure layout ------------------------------------------------------------
fig, (ax_ov, ax_zm) = plt.subplots(
    2, 1, figsize=(14, 7),
    gridspec_kw={"height_ratios": [1, 2]},
)
fig.suptitle("Perfusion Index - Overnight Recording", fontsize=13, fontweight="bold")
fig.subplots_adjust(hspace=0.40, left=0.07, right=0.97, top=0.93, bottom=0.09)

# -- Overview panel -----------------------------------------------------------
ax_ov.plot(t, p1, color=C1, lw=0.6, alpha=0.7, label="Unit 1")
ax_ov.plot(t, p2, color=C2, lw=0.6, alpha=0.7, label="Unit 2")
ax_ov.set_ylabel("PI (%)", fontsize=9)
ax_ov.set_title("Overview - drag to select zoom window", fontsize=9, color="#555555")
ax_ov.legend(loc="upper right", fontsize=8, framealpha=0.7)
ax_ov.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_ov.set_axisbelow(True)
ax_ov.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_ov.xaxis.set_major_locator(mdates.HourLocator())
plt.setp(ax_ov.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

# Highlight patch (hidden until first selection)
span_patch = ax_ov.axvspan(t.iloc[0], t.iloc[0], color="gold", alpha=0.3, zorder=0)
span_patch.set_visible(False)

# -- Zoom panel ---------------------------------------------------------------
ax_zm.plot(t, p1, color=C1, lw=1.0, label="Unit 1")
ax_zm.plot(t, p2, color=C2, lw=1.0, label="Unit 2")
ax_zm.set_ylabel("PI (%)", fontsize=10)
ax_zm.set_xlabel("Time", fontsize=10)
ax_zm.legend(loc="upper right", fontsize=9, framealpha=0.8)
ax_zm.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_zm.xaxis.grid(True, linestyle=":", alpha=0.3)
ax_zm.set_axisbelow(True)
ax_zm.set_title("Zoom view", fontsize=9, color="#555555")

def update_zoom(x0, x1):
    # Set xlim FIRST, then create a fresh locator that sees the constrained range.
    ax_zm.set_xlim(x0, x1)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax_zm.xaxis.set_major_locator(locator)
    ax_zm.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax_zm.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8)

# Initialise zoom to full range
update_zoom(t.iloc[0], t.iloc[-1])

# -- SpanSelector -------------------------------------------------------------
def on_select(xmin, xmax):
    x0 = mdates.num2date(xmin).replace(tzinfo=None)
    x1 = mdates.num2date(xmax).replace(tzinfo=None)
    if (x1 - x0).total_seconds() < 5:
        return
    span_patch.set_visible(True)
    span_patch.set_xy([[x0, 0], [x0, 1], [x1, 1], [x1, 0], [x0, 0]])
    span_patch.set_transform(ax_ov.get_xaxis_transform())
    update_zoom(x0, x1)
    mask = (t >= x0) & (t <= x1)
    if mask.any():
        ymin = min(p1[mask].min(), p2[mask].min())
        ymax = max(p1[mask].max(), p2[mask].max())
        margin = max((ymax - ymin) * 0.1, 0.2)
        ax_zm.set_ylim(ymin - margin, ymax + margin)
        ax_zm.set_title(
            f"Zoom  {x0.strftime('%H:%M:%S')} - {x1.strftime('%H:%M:%S')}",
            fontsize=9, color="#555555",
        )
    fig.canvas.draw_idle()

span_sel = SpanSelector(
    ax_ov, on_select, "horizontal",
    useblit=True,
    props={"alpha": 0.25, "facecolor": "gold"},
    interactive=True,
    drag_from_anywhere=True,
)

# -- Double-click to reset ----------------------------------------------------
def on_dblclick(event):
    if event.inaxes == ax_ov and event.dblclick:
        update_zoom(t.iloc[0], t.iloc[-1])
        ymin = min(p1.min(), p2.min())
        ymax = max(p1.max(), p2.max())
        ax_zm.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05)
        ax_zm.set_title("Zoom view", fontsize=9, color="#555555")
        span_patch.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_dblclick)

fig.text(
    0.5, 0.005,
    "Drag on overview to zoom  |  Double-click overview to reset",
    ha="center", va="bottom", fontsize=7.5, color="#777777",
)

plt.show()

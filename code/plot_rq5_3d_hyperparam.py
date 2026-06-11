"""
3D bar charts for Table rq5 (long-term + immediate metrics) on KuaiRand-Pure.
Run: python plot_rq5_3d_hyperparam.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

C_vals = np.array([9, 19, 29, 39, 49])
beta_vals = np.array([2, 4, 6, 8])          # front → back (Y-axis, low → high)

# ── Long-term metrics ─────────────────────────────────────────────────────────
RETURN_TIME = np.array(
    [
        [1.562, 1.512, 1.414, 1.498, 1.542],   # β=2
        [1.426, 1.487, 1.511, 1.476, 1.458],   # β=4
        [1.487, 1.467, 1.496, 1.512, 1.455],   # β=6
        [1.487, 1.465, 1.461, 1.497, 1.512],   # β=8
    ]
)

RETENTION = np.array(
    [
        [0.166, 0.168, 0.171, 0.170, 0.167],   # β=2
        [0.158, 0.157, 0.164, 0.165, 0.159],   # β=4
        [0.158, 0.162, 0.167, 0.169, 0.161],   # β=6
        [0.158, 0.161, 0.157, 0.154, 0.155],   # β=8
    ]
)

# ── Immediate-feedback metrics ───────────────────────────────────────────────
CLICK_RATE = np.array(
    [
        [0.881, 0.904, 0.922, 0.913, 0.889],   # β=2
        [0.846, 0.872, 0.896, 0.901, 0.868],   # β=4
        [0.838, 0.861, 0.887, 0.892, 0.857],   # β=6
        [0.829, 0.852, 0.841, 0.818, 0.833],   # β=8
    ]
)

LIKE_RATE = np.array(
    [
        [0.901, 0.928, 0.941, 0.919, 0.907],   # β=2
        [0.884, 0.913, 0.936, 0.931, 0.895],   # β=4
        [0.872, 0.904, 0.929, 0.934, 0.889],   # β=6
        [0.861, 0.892, 0.881, 0.868, 0.876],   # β=8
    ]
)

# Best in both tables: β=2, C=29
BEST_BETA_IDX = 0
BEST_C_IDX = 2


def _plot_3d_bars(
    values: np.ndarray,
    zlabel: str,
    title: str,
    out_path: str,
    *,
    lower_is_better: bool,
    cmap_name: str = "RdYlBu_r",  # 蓝(低) → 浅黄/米(中) → 红(高)，与论文常见 3D 热力柱一致
    figsize: tuple[float, float] = (8.5, 6.5),
    axis_label_fontsize: float = 20,
    tick_fontsize: float = 18,
    best_fontsize: float = 25,
    title_fontsize: float = 18,
    panel_fontsize: float = 25,
    star_size: int = 25,
    axis_label_pad: float = 10,
    z_tick_pad: float = 14,
):
    n_beta, n_c = values.shape
    assert n_beta == len(beta_vals) and n_c == len(C_vals)

    vmin, vmax = float(values.min()), float(values.max())

    dx = dy = 0.65
    gap = 0.1
    xpos, ypos = [], []
    for bi in range(n_beta):
        for ci in range(n_c):
            xpos.append(ci * (dx + gap))
            ypos.append(bi * (dx + gap))

    xpos = np.array(xpos)
    ypos = np.array(ypos)
    vals_flat = values.reshape(-1)
    z_floor = vmin
    zpos = np.full_like(vals_flat, z_floor, dtype=float)
    dz = vals_flat - z_floor

    norm = plt.Normalize(vmin, vmax)
    cmap = colormaps[cmap_name]
    if lower_is_better:
        colors = cmap(1.0 - norm(vals_flat))
    else:
        colors = cmap(norm(vals_flat))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.95, edgecolor="0.4", linewidth=0.2)

    bx = BEST_C_IDX * (dx + gap)
    by = BEST_BETA_IDX * (dx + gap)
    bz_top = values[BEST_BETA_IDX, BEST_C_IDX]
    ax.plot(
        [bx + dx / 2],
        [by + dy / 2],
        [bz_top + (vmax - vmin) * 0.06],
        "r*",
        markersize=star_size,
        zorder=10,
    )

    # best_txt = f"Best: ($\\beta$={beta_vals[BEST_BETA_IDX]}, $C$={C_vals[BEST_C_IDX]})"
    # ax.text2D(0.98, 0.92, best_txt, transform=ax.transAxes, fontsize=best_fontsize, color="darkorange", fontweight="bold", ha="right")

    ax.set_xlabel(r"$C$", labelpad=axis_label_pad, fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$\beta$", labelpad=axis_label_pad, fontsize=axis_label_fontsize)
    # ax.set_zlabel(zlabel, labelpad=axis_label_pad, fontsize=axis_label_fontsize)

    ax.set_xticks(np.arange(n_c) * (dx + gap) + dx / 2)
    ax.set_xticklabels([str(c) for c in C_vals], fontsize=tick_fontsize)
    ax.set_yticks(np.arange(n_beta) * (dx + gap) + dy / 2)
    ax.set_yticklabels([str(b) for b in beta_vals], fontsize=tick_fontsize)
    ax.set_zticks(np.linspace(vmin, vmax, 5))
    ax.set_zticklabels([f"{z:.3g}" for z in np.linspace(vmin, vmax, 5)], fontsize=tick_fontsize)
    # Z 轴刻度数字与轴线拉开距离（像素），避免与轴线重叠
    ax.tick_params(axis="z", pad=z_tick_pad)

    span = max(vmax - vmin, 1e-9)
    zpad_lo = max(span * 0.06, 1e-6)
    zpad_hi = max(span * 0.18, 1e-6)
    ax.set_zlim(z_floor - zpad_lo, vmax + zpad_hi)

    ax.view_init(elev=28, azim=-55)
    ax.grid(True, alpha=0.35)
    plt.title(title, fontsize=title_fontsize, pad=14)

    # Panel label
    # panel = "(a)" if "Return Time" in title else "(b)" if "Retention" in title else "(c)" if "Click" in title else "(d)"
    # ax.text2D(0.02, 0.98, panel, transform=ax.transAxes, fontsize=panel_fontsize, fontweight="bold", va="top")

    out_path = out_path.replace(".png", ".pdf")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    base = os.path.join(os.path.dirname(__file__), "output", "Kuairand_Pure", "agents")

    # Long-term
    _plot_3d_bars(RETURN_TIME, "Return Time",
                  r"(a) Return Time ($\mathrm{GFN4Retention}_{\beta,C}$, lower is better)",
                  os.path.join(base, "rq5_return_time_3d.png"),
                  lower_is_better=True, cmap_name="RdYlBu_r")

    _plot_3d_bars(RETENTION, "Retention",
                  r"(b) Retention ($\mathrm{GFN4Retention}_{\beta,C}$, higher is better)",
                  os.path.join(base, "rq5_retention_3d.png"),
                  lower_is_better=False, cmap_name="RdYlBu_r")

    # Immediate-feedback
    _plot_3d_bars(CLICK_RATE, "Click Rate",
                  r"(c) Click Rate ($\mathrm{GFN4Retention}_{\beta,C}$, higher is better)",
                  os.path.join(base, "rq5_click_rate_3d.png"),
                  lower_is_better=False, cmap_name="RdYlBu_r")

    _plot_3d_bars(LIKE_RATE, "Like Rate",
                  r"(d) Like Rate ($\mathrm{GFN4Retention}_{\beta,C}$, higher is better)",
                  os.path.join(base, "rq5_like_rate_3d.png"),
                  lower_is_better=False, cmap_name="RdYlBu_r")


if __name__ == "__main__":
    main()

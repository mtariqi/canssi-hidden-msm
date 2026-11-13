"""Figures. All figures are produced from synthetic data or aggregate outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAL = {"proposed": "#1b6ca8", "hidden_ni": "#e08a1e", "naive": "#8c8c8c",
       "observed": "#333333"}
LABELS = {"proposed": "Proposed (informative visits + RE)",
          "hidden_ni": "Hidden MSM, non-informative visits",
          "naive": "Naive observed-state MSM"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def state_diagram(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    boxes = {0: (1.2, 4.2, "Community\nstable"), 1: (1.2, 1.2, "Community\nunstable"),
             2: (6.0, 4.2, "Hospital\nstable"), 3: (6.0, 1.2, "Hospital\nunstable"),
             4: (8.9, 2.7, "Death")}
    for k, (x, y, lab) in boxes.items():
        col = "#f2dede" if k == 4 else ("#dbe9f6" if k in (0, 2) else "#fde7c8")
        ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.5), 1.9, 1.0, boxstyle="round,pad=0.05",
                                    fc=col, ec="#333"))
        ax.text(x, y, lab, ha="center", va="center", fontsize=9)

    def arr(a, b, txt, dx=0.0, dy=0.0, rad=0.0):
        (x1, y1, _), (x2, y2, _) = boxes[a], boxes[b]
        ax.annotate("", xy=(x2 + dx, y2 + dy), xytext=(x1 + dx, y1 + dy),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#333",
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=28, shrinkB=28))
        ax.text((x1 + x2) / 2 + dx * 2.2, (y1 + y2) / 2 + dy * 2.2, txt, fontsize=7.5,
                ha="center", va="center", color="#1b6ca8")
    arr(0, 1, "onset", dx=-0.25); arr(1, 0, "recovery", dx=0.25)
    arr(2, 3, "onset", dx=-0.25); arr(3, 2, "recovery", dx=0.25)
    arr(0, 2, "admission", dy=0.2); arr(2, 0, "discharge", dy=-0.2)
    arr(1, 3, "admission", dy=0.2); arr(3, 1, "discharge", dy=-0.2)
    for s in range(4):
        arr(s, 4, "", rad=0.0)
    ax.text(5, 5.65, "Observed: care setting C(t) and death     Latent: instability Z(t)",
            ha="center", fontsize=8.5, style="italic")
    ax.text(1.2, 0.2, "Encounters ~ Poisson(λ0·exp(w'γ + αZ + ρb)); emissions Y_m | Z at each encounter",
            fontsize=7.5, ha="left", color="#555")
    _save(fig, path)


def cohort_flow(flow, path):
    fig, ax = plt.subplots(figsize=(6.8, 0.62 * len(flow) + 0.4))
    ax.axis("off")
    n = len(flow)
    for i, (step, cnt) in enumerate(zip(flow.step, flow.n)):
        y = n - i
        ax.add_patch(FancyBboxPatch((0.05, y - 0.38), 0.9, 0.7, boxstyle="round,pad=0.02",
                                    fc="#dbe9f6" if i < n - 1 else "#1b6ca8", ec="#333",
                                    transform=ax.transData))
        ax.text(0.5, y, f"{step.strip()}\nN = {cnt:,}", ha="center", va="center", fontsize=8,
                color="white" if i == n - 1 else "black")
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - 0.62), xytext=(0.5, y - 0.38),
                        arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0.4, n + 0.6)
    ax.set_title("Cohort flow (SYNTHETIC data)")
    _save(fig, path)


def sim_performance(summ, quantity, path, title):
    """Bias and coverage by scenario for one estimand, with MC standard errors."""
    d = summ[summ.quantity == quantity].copy()
    scen = sorted(d.scenario.unique())
    methods = [m for m in ["proposed", "hidden_ni", "naive"] if m in d.method.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.33 * len(scen) + 1.4), sharey=True)
    y = np.arange(len(scen))
    off = np.linspace(-0.25, 0.25, len(methods))
    for j, m in enumerate(methods):
        dm = d[d.method == m].set_index("scenario").reindex(scen)
        axes[0].errorbar(dm.bias, y + off[j], xerr=1.96 * dm.mcse_bias, fmt="o", ms=3.5,
                         color=PAL[m], label=LABELS[m], capsize=0, lw=1)
        axes[1].errorbar(dm.coverage, y + off[j], xerr=1.96 * dm.mcse_coverage, fmt="o",
                         ms=3.5, color=PAL[m], capsize=0, lw=1)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[1].axvline(0.95, color="k", lw=0.8, ls="--")
    axes[0].set_yticks(y); axes[0].set_yticklabels(scen, fontsize=7)
    axes[0].set_xlabel("Bias (estimate − truth) ± 1.96 MCSE")
    axes[1].set_xlabel("95% interval coverage ± 1.96 MCSE")
    axes[1].set_xlim(-0.02, 1.02)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.12 - 0.3 / len(scen) * 8), ncol=3,
                   fontsize=7, frameon=False)
    fig.suptitle(title, fontweight="bold", fontsize=10)
    _save(fig, path)


def runtime_plot(fails, path):
    d = fails.copy()
    d["n"] = d.scenario.str.extract(r"n(\d+)_").astype(float)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in ["proposed", "hidden_ni", "naive"]:
        dm = d[d.method == m].groupby("n").mean_seconds.mean()
        ax.plot(dm.index, dm.values, "o-", color=PAL[m], label=LABELS[m])
    ax.set_xlabel("Cohort size"); ax.set_ylabel("Mean seconds per fit (1 CPU core)")
    ax.set_title("Computational cost of estimation")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def trajectories(t, probs, path, labels):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cols = ["#9ecae1", "#fdae6b", "#3182bd", "#e6550d", "#636363"]
    ax.stackplot(t, probs.T, labels=labels, colors=cols, alpha=0.95)
    ax.set_xlim(0, 365); ax.set_ylim(0, 1)
    ax.set_xlabel("Days since index discharge"); ax.set_ylabel("State occupancy probability")
    ax.set_title("Standardised state-occupancy probabilities (posterior mean)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5, frameon=False)
    _save(fig, path)


def calibration(ax, pred, obs, n, label, color):
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    se = np.sqrt(np.clip(obs * (1 - obs), 1e-6, None) / np.maximum(n, 1))
    ax.errorbar(pred, obs, yerr=1.96 * se, fmt="o-", ms=3.5, color=color, lw=1, label=label)


def forest(df, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(df) + 1.0))
    y = np.arange(len(df))[::-1]
    ax.errorbar(df.hr, y, xerr=[df.hr - df.lo, df.hi - df.hr], fmt="s", ms=4, color="#1b6ca8", lw=1)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(df.label, fontsize=7.5)
    ax.set_xlabel("Hazard / rate ratio (95% credible interval)")
    ax.set_title(title)
    _save(fig, path)

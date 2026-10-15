"""Make the paper figures from results/*.json (run after the experiments).

Colors: first three slots of the validated categorical palette (blue, orange, aqua). Each series also
has its own marker and line style, so the figures read in grayscale and for color-blind readers.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d6", "lines.linewidth": 1.4, "savefig.bbox": "tight",
    "pdf.fonttype": 42})
INK, MUTED = "#0b0b0b", "#6b6a66"
STYLE = {"MLP": dict(color="#2a78d6", marker="o", ls="-", label="MLP (data only)"),
         "PINN": dict(color="#eb6834", marker="s", ls="--", label="PINN (physics in the loss)"),
         "Hybrid": dict(color="#1baf7a", marker="^", ls="-.", label="Hybrid residual")}


def style_axes(ax):
    ax.grid(True, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, width=0.6, length=2.5)


def example1():
    r = json.loads((HERE / "results" / "example1_results.json").read_text())
    ns = r["config"]["n_train"]
    summ = r["summary_mean_std"]

    # ---- Figure: RMSE versus number of training samples
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharex=True)
    for ax, key, title in zip(axes, ("interpolation", "extrapolation"),
                              ("(a) Inside the training region", "(b) Outside the training region")):
        for kind, st in STYLE.items():
            # marker = mean over seeds (as in the text); bar = range from the best to the worst seed
            m = np.array([summ[f"{kind}_n{n}"][key][0] for n in ns])
            per = [[x[f"rmse_{key}"] for x in r["runs"] if x["model"] == kind and x["n"] == n] for n in ns]
            lo, hi = m - np.array([min(v) for v in per]), np.array([max(v) for v in per]) - m
            ax.errorbar(ns, m, yerr=[lo, hi], color=st["color"], marker=st["marker"], ls=st["ls"], ms=4,
                        capsize=1.5, elinewidth=0.6, label=st["label"])
        ax.axhline(r["nominal_physics_only_rmse_db"][key], color=MUTED, ls=":", lw=1.3, label="Nominal physics only")
        ax.axhline(r["config"]["noise_db"], color="#a9a8a3", ls=(0, (1, 4)), lw=0.9,
                   label=f"Noise std. ({r['config']['noise_db']:g} dB)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Training samples"); ax.set_title(title, pad=3)
        style_axes(ax)
    axes[0].set_ylabel("RMSE (dB)")
    h, lab = axes[1].get_legend_handles_labels()
    order = [lab.index(STYLE[k]["label"]) for k in STYLE] + [i for i, x in enumerate(lab) if x not in [v["label"] for v in STYLE.values()]]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.29), handlelength=2.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig_example1_rmse.pdf"); plt.close(fig)

    # ---- Figure: transmission loss versus distance at two frequencies
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=False)
    tr = r["curves"]["train_points_n30_seed0"]
    for ax, fk, title in zip(axes, ("f20", "f35"), ("(a) $f$ = 20 kHz (in training band)", "(b) $f$ = 35 kHz (outside)")):
        c = r["curves"][fk]
        l = np.array(c["l"])
        if fk == "f20":   # 20 kHz lies inside the training band, so shade the training distances
            ax.axvspan(l.min(), r["config"]["train_region"]["l_max_km"], color="#efefec", lw=0)
        ax.plot(l, c["truth"], color=INK, lw=1.8, label="True channel")
        ax.plot(l, c["nominal"], color=MUTED, ls=":", lw=1.0, label="Nominal physics")
        for kind, st in STYLE.items():
            ax.plot(l, c[kind], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, markevery=30, label=st["label"])
        if fk == "f20":
            fl, yl = np.array(tr["f"]), np.array(tr["y"])
            sel = np.abs(fl - 20) < 3
            ax.scatter(np.array(tr["l"])[sel], yl[sel], s=8, facecolors="none", edgecolors=INK, lw=0.6, zorder=5, label="Training samples (17\u201323 kHz)")
        ax.set_xlabel("Distance (km)"); ax.set_title(title, pad=3)
        style_axes(ax)
    axes[0].set_ylabel("Transmission loss (dB)")
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.25), handlelength=2.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig_example1_curves.pdf"); plt.close(fig)
    print("Example 1 figures written")


def example2():
    """ROC curves of the physics-residual detector and the MLP detector (seed 0) for the four test cases."""
    r = json.loads((HERE / "results" / "example2_results.json").read_text())
    cases = [("a_seen", "(a) Seen attack, training deployment"), ("b_env_shift", "(b) Seen attack, new deployment"),
             ("c_unseen_attack", "(c) Unseen two-link attack"), ("d_consistent_spoof", "(d) Physics-consistent spoof")]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3), sharex=True, sharey=True)
    for ax, (k, title) in zip(axes.ravel(), cases):
        ax.plot([0, 1], [0, 1], color="#a9a8a3", ls=":", lw=0.9, label="Chance")
        fp, tp = r["roc"][k]["physics"]
        ax.plot(fp, tp, color="#eb6834", ls="--", lw=1.5, label="Physics residual")
        fm, tm = r["roc"][k]["mlp"]
        ax.plot(fm, tm, color="#2a78d6", ls="-", lw=1.5, label="MLP (data only)")
        a_p = r["physics"][k]["auc"]; a_m = r["mlp_summary"][k]["auc"][0]
        ax.text(0.97, 0.05, f"AUC {a_p:.2f} / {a_m:.2f}", ha="right", va="bottom", fontsize=6.5, color=INK)
        ax.set_title(title, pad=2, fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
        style_axes(ax)
    for ax in axes[1]: ax.set_xlabel("False-alarm rate")
    for ax in axes[:, 0]: ax.set_ylabel("Detection rate")
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04), handlelength=2.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(FIG / "fig_example2_roc.pdf"); plt.close(fig)
    print("Example 2 figure written")


def trend():
    """Fig. 7: PIML papers per year and segment, counted from the screening table (role 'PIML primary')."""
    import openpyxl
    wb = openpyxl.load_workbook(HERE.parent / "data" / "screening_table.xlsx", read_only=True)
    ws = wb["Screening"]
    rows = list(ws.iter_rows(values_only=True)); hdr = rows[0]
    iR, iS, iY = hdr.index("Role"), hdr.index("Segment"), hdr.index("Year")
    segs = ["Space", "Air", "Ground", "Sea surface", "Underwater"]
    years = list(range(2019, 2027))
    cnt = {s: [sum(1 for x in rows[1:] if x[iR] == "PIML primary" and x[iS] == s and int(x[iY]) == y) for y in years] for s in segs}
    # Okabe-Ito colors (color-blind safe) plus a hatch per segment for grayscale print
    col = {"Space": "#0072B2", "Air": "#56B4E9", "Ground": "#009E73", "Sea surface": "#E69F00", "Underwater": "#CC79A7"}
    hat = {"Space": "", "Air": "////", "Ground": "", "Sea surface": "\\\\\\\\", "Underwater": "...."}
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    bottom = np.zeros(len(years))
    for s in segs:
        v = np.array(cnt[s])
        ax.bar(years, v, bottom=bottom, width=0.72, color=col[s], edgecolor="white", linewidth=0.6, hatch=hat[s], label=s)
        bottom += v
    for x, t in zip(years, bottom):
        if t: ax.text(x, t + 0.6, f"{int(t)}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(years); ax.set_xticklabels([str(y) if y < 2026 else "2026*" for y in years])
    ax.set_ylabel("Papers"); ax.set_ylim(0, bottom.max() * 1.15)
    style_axes(ax); ax.grid(False, axis="x")
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=6.5, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig_trend.pdf"); plt.close(fig)
    json.dump({"years": years, "counts": cnt}, open(HERE / "results" / "trend_counts.json", "w"), indent=1)
    print("Trend figure written", {s: sum(v) for s, v in cnt.items()})


if __name__ == "__main__":
    example1()
    if (HERE / "results" / "example2_results.json").exists():
        example2()
    trend()

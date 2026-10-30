#!/usr/bin/env python3
"""Publication-grade ARC-SCEB interpretability figures.

Each figure targets one ARC-SCEB design diagnostic:
  Fig 1  modality-structured expert use    -> why audit video first
  Fig 2  depth-dependent router separation -> audit-selected layer set
  Fig 3  router-confidence risk & budget    -> boundary rule -> top-4/6/8
  Fig 4  audio cross-modal control          -> same-sample audio floor (6/8)
  Fig 5  method schematic / decision flow   -> one-glance mechanism summary

============================  EDITING GUIDE (for paper writing)  =============
Inputs (produced once; no GPU needed to re-plot):
  <repo>/logs/arc_sceb_interpretability/aggregated_stats.json  (998 OVB + 1495 AVUT)
  <repo>/logs/arc_sceb_interpretability/worker_capture.json    (real prefill replay)

Run:
  python3 scripts/arc_sceb_make_figures.py                 # re-render all 5 figures
  python3 scripts/arc_sceb_make_figures.py --only fig3     # just one figure
  python3 scripts/arc_sceb_make_figures.py --only fig3 fig4
  python3 scripts/arc_sceb_make_figures.py --outdir /tmp/figs --formats pdf svg
  python3 scripts/arc_sceb_make_figures.py --repo /path/to/Omni-MoE-Pruner

Where to edit for the paper:
  * Colours / palette ............ dict ``C`` and ``STAGE_BAND`` below.
  * Global look (fonts, dpi, ...) . ``mpl.rcParams`` block below.
  * Figure size .................. ``plt.subplots(..., figsize=(W, H))`` in each figX_*().
  * Panel titles / axis labels ... ``ax.set_title`` / ``ax.set_xlabel`` in each figX_*().
  * Sub-caption "(a)/(b)" ........ the ``loc="left"`` titles in each panel.
  * Suptitle (bold header) ....... ``fig.suptitle(...)`` at the end of each figX_*().
  * Layer window / thresholds .... read from the JSON ``policy`` block, not hard-coded.
  * Output formats/dpi ........... ``--formats`` and rcParams["savefig.dpi"].

Each ``figN_*`` function is self-contained: to tweak one figure, edit only that
function and re-run with ``--only figN``. Vector (PDF/SVG) output uses embedded
fonts (Type-42) so text stays editable in Illustrator / Inkscape.
============================================================================="""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
_MPL_CONFIG = Path(os.environ.get("TMPDIR", "/tmp")) / "arc_sceb_matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

def _find_default_repo():
    """Support both ``scripts/`` and the archived ``plotting/`` copy."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "moe_expert_budget").is_dir() and (parent / "logs").is_dir():
            return parent
    raise RuntimeError("Cannot locate Omni-MoE-Pruner repository root")


_DEFAULT_REPO = _find_default_repo()
REPO = _DEFAULT_REPO
INTERP = REPO / "logs/arc_sceb_interpretability"
FIG = INTERP / "figures"
FORMATS = ["pdf", "png"]  # overridable with --formats


def _configure(repo=None, outdir=None, formats=None):
    """Resolve input/output locations (call once from main)."""
    global REPO, INTERP, FIG, FORMATS
    if repo:
        REPO = Path(repo).resolve()
    INTERP = REPO / "logs/arc_sceb_interpretability"
    FIG = Path(outdir).resolve() if outdir else (INTERP / "figures")
    if formats:
        FORMATS = list(formats)
    FIG.mkdir(parents=True, exist_ok=True)

# ---- global publication style (ICLR-like, colorblind-safe) ----------------
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.3,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Okabe-Ito colorblind-safe palette
C = {
    "video": "#0072B2",   # blue
    "audio": "#D55E00",   # vermillion
    "text":  "#009E73",   # green
    "image": "#CC79A7",   # purple
    "m4":    "#0072B2",
    "m6":    "#E69F00",   # orange
    "m8":    "#999999",   # grey
    "accent":"#D55E00",
    "guard": "#D55E00",
    "mid":   "#0072B2",
}
STAGE_BAND = "#FDE9D9"  # soft highlight for middle layers


def load():
    agg = json.loads((INTERP / "aggregated_stats.json").read_text())
    cap_path = INTERP / "worker_capture.json"
    cap = json.loads(cap_path.read_text()) if cap_path.exists() else None
    return agg, cap


def arr(stat_block, key):
    m = stat_block[key]["mean"]
    lo = stat_block[key]["p25"]
    hi = stat_block[key]["p75"]
    f = lambda xs: np.array([x if x is not None else np.nan for x in xs], dtype=float)
    return f(m), f(lo), f(hi)


# ============================================================= Figure 1
def fig1_modality_dispersion(agg):
    """Describe modality-level expert-use dispersion without claiming safety."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    benches = [("omnivideobench", "OmniVideoBench"), ("avut", "AVUT")]

    # (a) grouped bars: sample-level middle-layer expert-use entropy
    ax = axes[0]
    mods = ["video", "audio", "text"]
    xpos = np.arange(len(benches))
    w = 0.25
    for j, mod in enumerate(mods):
        vals, errlo, errhi = [], [], []
        for bk, _ in benches:
            s = agg[bk]
            stat = s["middle_entropy"][mod]
            vals.append(stat["mean"])
            errlo.append(stat["mean"] - stat["p25"])
            errhi.append(stat["p75"] - stat["mean"])
        ax.bar(xpos + (j - 1) * w, vals, w, yerr=[errlo, errhi], capsize=2.5,
               color=C[mod], label=mod, edgecolor="white", linewidth=0.5,
               error_kw={"elinewidth": 0.8})
    ax.set_xticks(xpos)
    ax.set_xticklabels([b[1] for b in benches])
    ax.set_ylabel("Normalized expert-use entropy\n(middle layers)")
    ax.set_ylim(0.80, 0.96)
    ax.set_title("(a) Video routing is most diffuse", loc="left", fontweight="bold")
    ax.legend(title="modality", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=7.5, columnspacing=1.0, handlelength=1.2)

    # (b) experts needed to cover 50% of routing mass (video vs audio vs text)
    ax = axes[1]
    s = agg["omnivideobench"]
    layers = np.array(s["layers"])
    for mod in mods:
        m, lo, hi = arr(s, f"spread50_{mod}")
        ax.plot(layers, m * 128, color=C[mod], label=mod)
        ax.fill_between(layers, lo * 128, hi * 128, color=C[mod], alpha=0.13, linewidth=0)
    L0, L1 = s["policy"]["L_START"], s["policy"]["L_END"]
    ax.axvspan(L0, L1 - 1, color=STAGE_BAND, zorder=0, label="candidate middle layers")
    ax.set_xlabel("Text-MoE layer index")
    ax.set_ylabel("Experts accounting for 50%\nof activations (of 128)")
    ax.set_title("(b) Video uses a broader expert set", loc="left", fontweight="bold")
    ax.legend(ncol=2, loc="lower center", fontsize=7, columnspacing=1.0, handlelength=1.4)
    ax.set_xlim(0, 47)
    ax.margins(y=0.14)

    fig.subplots_adjust(wspace=0.34, top=0.92)
    _save(fig, "fig1_modality_dispersion")


# ============================================================= Figure 2
def fig2_layer_stage(agg):
    """Describe depth-dependent router separation without claiming safety."""
    s = agg["omnivideobench"]
    layers = np.array(s["layers"])
    L0, L1 = s["policy"]["L_START"], s["policy"]["L_END"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)

    # (a) top1-top2 decisiveness margin by layer
    ax = axes[0]
    m, lo, hi = arr(s, "m12_mean")
    ax.axvspan(L0, L1 - 1, color=STAGE_BAND, zorder=0)
    ax.plot(layers, m * 1e3, color=C["video"])
    ax.fill_between(layers, lo * 1e3, hi * 1e3, color=C["video"], alpha=0.15, linewidth=0)
    ax.axvline(L0, color=C["accent"], ls="--", lw=1.0)
    ax.axvline(L1 - 1, color=C["accent"], ls="--", lw=1.0)
    ax.set_ylabel(r"top1$-$top2 margin ($\times10^{-3}$)")
    ax.set_xlabel("Text-MoE layer index")
    ax.set_title("(a) Top-candidate separation", loc="left", fontweight="bold", fontsize=9)
    ax.text((L0 + L1) / 2, ax.get_ylim()[1] * 0.96, "candidate\nwindow", ha="center",
            va="top", fontsize=7.5, color=C["accent"])

    # (b) selected-boundary margin by layer (mean with p25-p75)
    ax = axes[1]
    m, lo, hi = arr(s, "bnd_mean")
    ax.axvspan(L0, L1 - 1, color=STAGE_BAND, zorder=0, label="candidate window [12,32)")
    ax.plot(layers, m * 1e3, color=C["m6"], label="selected boundary margin")
    ax.fill_between(layers, lo * 1e3, hi * 1e3, color=C["m6"], alpha=0.15, linewidth=0)
    thr = s["policy"]["boundary_threshold"] * 1e3
    ax.axhline(thr, color=C["accent"], ls=":", lw=1.2, label=f"diagnostic $\\tau_b$={thr:.3f}e-3")
    ax.set_xlabel("Text-MoE layer index")
    ax.set_ylabel(r"boundary margin ($\times10^{-3}$)")
    ax.set_title("(b) Candidate-boundary separation", loc="left", fontweight="bold", fontsize=9)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 47)

    fig.subplots_adjust(wspace=0.30, top=0.92)
    _save(fig, "fig2_layer_stage")


def _collect_capture(cap, key, layer_lo=None, layer_hi=None):
    """Concatenate a raw per-layer array across all captured samples/layers."""
    out = []
    for s in cap["samples"]:
        for L in s["per_layer"]:
            if layer_lo is not None and not (layer_lo <= L["layer"] < layer_hi):
                continue
            out.extend(L.get(key, []))
    return np.array(out, dtype=float)


# ============================================================= Figure 3
def fig3_confidence_budget(agg, cap):
    """Boundary margin geometry drives the per-token top-4/6/8 budget."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    thr = agg["omnivideobench"]["policy"]["boundary_threshold"]

    # (a) distribution of video-token m=4 boundary margin (p4-p5) w/ threshold
    ax = axes[0]
    if cap is not None:
        vb = _collect_capture(cap, "raw_video_boundary_small",
                              cap["policy"]["L_START"], cap["policy"]["L_END"])
        vb = vb[np.isfinite(vb)]
        vb_c = np.clip(vb, 0, np.percentile(vb, 99))
        ax.hist(vb_c * 1e3, bins=60, color=C["video"], alpha=0.85, edgecolor="white", linewidth=0.3)
        below = float((vb < thr).mean()) * 100
        ax.axvline(thr * 1e3, color=C["accent"], ls="--", lw=1.4)
        ax.annotate(f"$\\tau$={thr*1e3:.3f}e-3", xy=(thr * 1e3, ax.get_ylim()[1] * 0.9),
                    xytext=(thr * 1e3 + 0.6, ax.get_ylim()[1] * 0.9), color=C["accent"], fontsize=8)
        ax.text(0.97, 0.55,
                f"{below:.0f}% below $\\tau$\n$\\rightarrow$ stay $\\geq$top-6\n\n"
                f"{100-below:.0f}% above $\\tau$\n$\\rightarrow$ eligible top-4",
                transform=ax.transAxes, ha="right", va="center", fontsize=7.5)
        src = "real Qwen3-Omni prefill"
    else:
        src = "capture pending"
    ax.set_xlabel(r"video-token boundary margin $p_4-p_5$ ($\times10^{-3}$)")
    ax.set_ylabel("count (video token-layers)")
    ax.set_title("(a) Gap gates top-4 eligibility", loc="left", fontweight="bold")

    # (b) effective budget histogram top-4/6/8 (share) from real replay
    ax = axes[1]
    if cap is not None:
        tot = {4: 0, 6: 0, 8: 0}
        for s in cap["samples"]:
            for k, v in s["chosen_m_hist_total"].items():
                tot[int(k)] += int(v)
        n = sum(tot.values()) or 1
        shares = [tot[4] / n * 100, tot[6] / n * 100, tot[8] / n * 100]
        bars = ax.bar(["top-4", "top-6", "top-8"], shares,
                      color=[C["m4"], C["m6"], C["m8"]], edgecolor="white", linewidth=0.6)
        for b, sh in zip(bars, shares):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{sh:.1f}%",
                    ha="center", fontsize=8)
        eff = (4 * tot[4] + 6 * tot[6] + 8 * tot[8]) / n
        ax.set_ylim(0, max(shares) * 1.25)
        ax.set_ylabel("Share of video token-layer masks (%)")
        ax.set_title(f"(b) Dynamic mask budgets (mean {eff:.2f})",
                     loc="left", fontweight="bold")
        ax.annotate("selective, not uniform\ntop-k reduction", xy=(0.5, 0.82),
                    xycoords="axes fraction", ha="center", fontsize=7.5, style="italic",
                    color=C["video"])
    fig.subplots_adjust(wspace=0.30, top=0.92)
    _save(fig, "fig3_confidence_budget")


# ============================================================= Figure 4
def fig4_audio_control(agg, cap):
    """Same-sample audio-router margin gates the video floor (the ARC novelty)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    if cap is None:
        fig.suptitle("Fig. 4  Audio cross-modal control (capture pending)", y=1.02)
        _save(fig, "fig4_audio_control")
        return
    frag = cap["policy"]["fragile_margin"]

    # (a) distribution of the ACTUAL gating variable: per-(sample,layer)
    #     same-sample audio decisiveness mean(p1-p2). Split-coloured by the
    #     fragility threshold tau -> shares the same x-axis as panel (b).
    ax = axes[0]
    g, gstate = [], []
    for s in cap["samples"]:
        for L in s["per_layer"]:
            g.append(L["audio_margin12_mean"] * 1e3)
            gstate.append(L["audio_state"])
    g = np.array(g)
    gt = frag * 1e3
    gstate = np.asarray(gstate)
    fragile_mask = gstate == "fragile_audio"
    confident_mask = gstate == "confident_audio"
    insufficient_mask = gstate == "insufficient_audio"
    n_frag = int(fragile_mask.sum())
    n_conf = int(confident_mask.sum())
    n_ins = int(insufficient_mask.sum())
    edges = np.arange(np.floor(g.min()), np.ceil(g.max()) + 1.0, 1.0)
    ax.hist(g[fragile_mask], bins=edges, color=C["guard"], alpha=0.9,
            edgecolor="white", linewidth=0.4, label=f"fragile: floor $\\geq$6 ({n_frag} states)")
    ax.hist(g[confident_mask], bins=edges, color=C["text"], alpha=0.85,
            edgecolor="white", linewidth=0.4, label=f"confident: unchanged ({n_conf} states)")
    if n_ins:
        ax.hist(g[insufficient_mask], bins=edges, color=C["m8"], alpha=0.85,
                edgecolor="white", linewidth=0.4,
                label=f"insufficient: floor 8 ({n_ins} states)")
    ax.margins(y=0.28)
    ax.axvline(gt, color="black", ls="--", lw=1.3)
    ax.annotate(f"fragility threshold $\\tau$ = {gt:.1f}e-3",
                xy=(gt, ax.get_ylim()[1] * 0.55), xytext=(gt + 2.6, ax.get_ylim()[1] * 0.55),
                fontsize=7.5, va="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel(r"same-sample audio decisiveness $\overline{p_1-p_2}$ ($\times10^{-3}$)")
    ax.set_ylabel("count (sample-layers)")
    ax.set_title("(a) Audio state triggers the guard", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7)

    # (b) per (sample,layer): audio margin -> fraction of video tokens floored
    ax = axes[1]
    xs, ys, cols = [], [], []
    state_col = {"confident_audio": C["text"], "fragile_audio": C["guard"], "insufficient_audio": C["m8"]}
    for s in cap["samples"]:
        for L in s["per_layer"]:
            h = L["chosen_m_hist"]
            tot = sum(int(x) for x in h.values()) or 1
            floored = (int(h.get("6", 0)) + int(h.get("8", 0))) / tot * 100
            xs.append(L["audio_margin12_mean"] * 1e3)
            ys.append(floored)
            cols.append(state_col.get(L["audio_state"], C["m8"]))
    ax.scatter(xs, ys, c=cols, s=26, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(frag * 1e3, color="black", ls="--", lw=1.2)
    ax.set_xlabel(r"same-sample audio decisiveness $\overline{p_1-p_2}$ ($\times10^{-3}$)")
    ax.set_ylabel(r"Video masks kept at top-6 or top-8 (%)")
    ax.set_title("(b) The guard only raises budgets", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=state_col[k], mec="white",
                      label=k.replace("_", " ")) for k in state_col]
    ax.legend(handles=handles, loc="center right", fontsize=6.8, handletextpad=0.3,
              borderaxespad=0.3)
    ax.set_ylim(25, 105)

    fig.subplots_adjust(wspace=0.30, top=0.92)
    _save(fig, "fig4_audio_control")


# ===================================================== Paper audit figure
def fig_paper_routing_audit(agg, cap):
    """Compact evidence figure for the paper's motivating study.

    Every panel is descriptive.  In particular, expert-use entropy is not
    labeled as redundancy, and replay token-layer masks are not presented as
    independent benchmark samples or as executed-compute savings.
    """
    if cap is None:
        raise FileNotFoundError(
            "worker_capture.json is required for the evidence-bearing paper figure"
        )

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.65))
    benches = [("omnivideobench", "OmniVideoBench"), ("avut", "AVUT")]
    mods = ["video", "audio", "text"]
    hatches = ["///", "\\\\", "..."]

    # (a) One middle-layer mean per sample; bars show mean and sample IQR.
    ax = axes[0, 0]
    xpos = np.arange(len(benches))
    width = 0.23
    for j, (mod, hatch) in enumerate(zip(mods, hatches)):
        means, lower, upper = [], [], []
        for bench, _ in benches:
            stat = agg[bench]["middle_entropy"][mod]
            means.append(stat["mean"])
            lower.append(stat["mean"] - stat["p25"])
            upper.append(stat["p75"] - stat["mean"])
        ax.bar(
            xpos + (j - 1) * width,
            means,
            width,
            yerr=[lower, upper],
            capsize=2,
            color=C[mod],
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=mod,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels([name for _, name in benches], rotation=8)
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Normalized expert-use entropy")
    ax.set_title("(a) Expert use differs by modality", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="lower center", columnspacing=0.7, handlelength=1.1)

    # (b) Candidate-boundary gap by depth.  The ribbon is the sample IQR.
    ax = axes[0, 1]
    s = agg["omnivideobench"]
    layers = np.asarray(s["layers"])
    gap, gap_lo, gap_hi = arr(s, "bnd_mean")
    L0, L1 = s["policy"]["L_START"], s["policy"]["L_END"]
    ax.axvspan(L0, L1 - 1, color=STAGE_BAND, zorder=0)
    ax.fill_between(
        layers, gap_lo * 1e3, gap_hi * 1e3,
        color=C["m6"], alpha=0.18, linewidth=0,
    )
    ax.plot(layers, gap * 1e3, color=C["m6"], linewidth=1.6)
    threshold = s["policy"]["boundary_threshold"] * 1e3
    ax.axhline(threshold, color=C["accent"], linestyle=":", linewidth=1.0)
    ax.text(
        46.5, threshold + 0.012, r"$\tau_b$", ha="right", va="bottom",
        color=C["accent"], fontsize=7,
    )
    ax.text(
        (L0 + L1 - 1) / 2, 0.175, "candidate layers 12--31",
        ha="center", va="bottom", color=C["accent"], fontsize=7,
    )
    ax.set_xlim(0, 47)
    ax.set_ylim(0.16, 0.73)
    ax.set_xlabel("Text-MoE layer index")
    ax.set_ylabel(r"Candidate-boundary gap ($\times10^{-3}$)")
    ax.set_title("(b) Separation varies with depth", loc="left", fontweight="bold")

    # (c) Dynamic mask decisions pooled over the eight replay clips.
    ax = axes[1, 0]
    totals = {4: 0, 6: 0, 8: 0}
    for sample in cap["samples"]:
        for key, value in sample["chosen_m_hist_total"].items():
            totals[int(key)] += int(value)
    total_masks = sum(totals.values())
    shares = np.asarray([totals[k] / total_masks * 100 for k in (4, 6, 8)])
    bars = ax.bar(
        ["top-4", "top-6", "top-8"], shares,
        color=[C["m4"], C["m6"], C["m8"]],
        edgecolor="#555555", linewidth=0.5,
        hatch=["///", "\\\\", "..."],
    )
    for bar, share in zip(bars, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f"{share:.1f}%", ha="center", va="bottom", fontsize=7,
        )
    mean_budget = sum(k * totals[k] for k in totals) / total_masks
    ax.text(
        0.98, 0.93, f"mean = {mean_budget:.2f}",
        transform=ax.transAxes, ha="right", va="top", color=C["video"],
        fontsize=7, fontstyle="italic",
    )
    ax.set_ylim(0, 62)
    ax.set_ylabel("Share of video token-layer masks (%)")
    ax.set_title("(c) Dynamic policy retains mixed budgets", loc="left", fontweight="bold")

    # (d) One point per (sample, middle-layer) state; 160 states in this replay.
    ax = axes[1, 1]
    state_data = {
        "confident_audio": ([], []),
        "fragile_audio": ([], []),
        "insufficient_audio": ([], []),
    }
    for sample in cap["samples"]:
        for layer in sample["per_layer"]:
            hist = {int(k): int(v) for k, v in layer["chosen_m_hist"].items()}
            denom = sum(hist.values()) or 1
            kept = (hist.get(6, 0) + hist.get(8, 0)) / denom * 100
            xs, ys = state_data[layer["audio_state"]]
            xs.append(layer["audio_margin12_mean"] * 1e3)
            ys.append(kept)
    state_style = {
        "confident_audio": (C["text"], "o", "confident"),
        "fragile_audio": (C["guard"], "^", "fragile"),
        "insufficient_audio": (C["m8"], "s", "insufficient"),
    }
    for state in ("confident_audio", "fragile_audio", "insufficient_audio"):
        xs, ys = state_data[state]
        if not xs:
            continue
        color, marker, label = state_style[state]
        ax.scatter(
            xs, ys, s=17, marker=marker, color=color, alpha=0.78,
            edgecolor="white", linewidth=0.35,
            label=f"{label} ({len(xs)})", zorder=3,
        )
    audio_threshold = cap["policy"]["fragile_margin"] * 1e3
    ax.axvline(audio_threshold, color="black", linestyle="--", linewidth=0.9)
    ax.text(
        audio_threshold + 0.25, 103, r"$\tau_a$", ha="left", va="top", fontsize=7,
    )
    fragile_y = state_data["fragile_audio"][1]
    n_states = sum(len(v[0]) for v in state_data.values())
    if fragile_y and min(fragile_y) >= 99.999:
        ax.text(
            0.98, 0.92,
            f"{len(fragile_y)}/{n_states} fragile states\n"
            + r"all video masks $\geq$top-6",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.7,
            color=C["guard"],
        )
    ax.set_ylim(25, 105)
    ax.set_xlabel(r"Audio decisiveness $\overline{p_1-p_2}$ ($\times10^{-3}$)")
    ax.set_ylabel("Video masks at top-6 or top-8 (%)")
    ax.set_title("(d) Audio guard only raises budgets", loc="left", fontweight="bold")
    ax.legend(loc="center right", handletextpad=0.25, borderaxespad=0.2)

    fig.subplots_adjust(
        left=0.105, right=0.99, bottom=0.10, top=0.98,
        wspace=0.40, hspace=0.52,
    )
    _save(fig, "routing_audit")


def fig_paper_risk_map(agg):
    """Two-panel motivation figure: modality first, then depth."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    benches = [("omnivideobench", "OmniVideoBench"), ("avut", "AVUT")]
    mods = ["video", "audio", "text"]
    hatches = ["///", "\\\\", "..."]

    ax = axes[0]
    xpos = np.arange(len(benches))
    width = 0.23
    for j, (mod, hatch) in enumerate(zip(mods, hatches)):
        means, lower, upper = [], [], []
        for bench, _ in benches:
            stat = agg[bench]["middle_entropy"][mod]
            means.append(stat["mean"])
            lower.append(stat["mean"] - stat["p25"])
            upper.append(stat["p75"] - stat["mean"])
        ax.bar(
            xpos + (j - 1) * width,
            means,
            width,
            yerr=[lower, upper],
            capsize=2,
            color=C[mod],
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=mod,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels([name for _, name in benches], rotation=7)
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Normalized expert-use entropy")
    ax.set_title("(a) Video uses the broadest expert set", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="lower center", columnspacing=0.55, handlelength=1.0)

    ax = axes[1]
    s = agg["omnivideobench"]
    layers = np.asarray(s["layers"])
    gap, gap_lo, gap_hi = arr(s, "bnd_mean")
    L0, L1 = s["policy"]["L_START"], s["policy"]["L_END"]
    ax.axvspan(L0, L1 - 1, color=STAGE_BAND, zorder=0)
    ax.fill_between(
        layers, gap_lo * 1e3, gap_hi * 1e3,
        color=C["m6"], alpha=0.18, linewidth=0,
    )
    ax.plot(layers, gap * 1e3, color=C["m6"], linewidth=1.6)
    threshold = s["policy"]["boundary_threshold"] * 1e3
    ax.axhline(threshold, color=C["accent"], linestyle=":", linewidth=1.0)
    ax.text(46.5, threshold + 0.012, r"$\tau_b$", ha="right", va="bottom",
            color=C["accent"], fontsize=7)
    ax.text((L0 + L1 - 1) / 2, 0.175, "layers considered for reduction",
            ha="center", va="bottom", color=C["accent"], fontsize=7)
    ax.set_xlim(0, 47)
    ax.set_ylim(0.16, 0.73)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel(r"Cutoff gap ($\times10^{-3}$)")
    ax.set_title("(b) The cutoff gap changes across layers", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.94, wspace=0.38)
    _save(fig, "risk_map")


def fig_paper_modality_target(agg):
    """Standalone paper figure: modality routing dispersion."""
    fig, ax = plt.subplots(figsize=(4.65, 2.15))
    benches = [("omnivideobench", "OmniVideoBench"), ("avut", "AVUT")]
    mods = ["video", "audio", "text"]
    hatches = ["///", "\\\\", "..."]
    xpos = np.arange(len(benches))
    width = 0.23
    for j, (mod, hatch) in enumerate(zip(mods, hatches)):
        means, lower, upper = [], [], []
        for bench, _ in benches:
            stat = agg[bench]["middle_entropy"][mod]
            means.append(stat["mean"])
            lower.append(stat["mean"] - stat["p25"])
            upper.append(stat["p75"] - stat["mean"])
        ax.bar(
            xpos + (j - 1) * width,
            means,
            width,
            yerr=[lower, upper],
            capsize=2,
            color=C[mod],
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=mod,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels([name for _, name in benches])
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Normalized expert-use entropy")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.18, top=0.84)
    _save(fig, "modality_target")


def fig_paper_depth_profile(agg):
    """Standalone paper figure: Qwen3-Omni cutoff-gap depth profile."""
    fig, ax = plt.subplots(figsize=(4.65, 2.15))
    s = agg["omnivideobench"]
    layers = np.asarray(s["layers"])
    depth = layers / max(len(layers) - 1, 1)
    gap, gap_lo, gap_hi = arr(s, "bnd_mean")
    L0, L1 = s["policy"]["L_START"], s["policy"]["L_END"]
    d0 = L0 / max(len(layers) - 1, 1)
    d1 = (L1 - 1) / max(len(layers) - 1, 1)
    ax.axvspan(d0, d1, color=STAGE_BAND, zorder=0)
    ax.fill_between(
        depth, gap_lo * 1e3, gap_hi * 1e3,
        color=C["m6"], alpha=0.18, linewidth=0,
    )
    ax.plot(depth, gap * 1e3, color=C["m6"], linewidth=1.6)
    ax.text((d0 + d1) / 2, 0.175, "audit-selected layers",
            ha="center", va="bottom", color=C["accent"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.16, 0.73)
    ax.set_xlabel("Normalized MoE depth")
    ax.set_ylabel(r"Cutoff gap ($\times10^{-3}$)")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.98)
    _save(fig, "depth_profile")


def fig_paper_policy_behavior(agg, cap):
    """Two-panel experiment figure: selected budgets and audio protection."""
    if cap is None:
        raise FileNotFoundError("worker_capture.json is required for policy behavior")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))

    ax = axes[0]
    totals = {4: 0, 6: 0, 8: 0}
    for sample in cap["samples"]:
        for key, value in sample["chosen_m_hist_total"].items():
            totals[int(key)] += int(value)
    total_masks = sum(totals.values())
    shares = np.asarray([totals[k] / total_masks * 100 for k in (4, 6, 8)])
    bars = ax.bar(
        ["4 experts", "6 experts", "8 experts"], shares,
        color=[C["m4"], C["m6"], C["m8"]],
        edgecolor="#555555", linewidth=0.5, hatch=["///", "\\\\", "..."],
    )
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{share:.1f}%", ha="center", va="bottom", fontsize=7)
    mean_budget = sum(k * totals[k] for k in totals) / total_masks
    ax.text(0.98, 0.93, f"mean = {mean_budget:.2f}", transform=ax.transAxes,
            ha="right", va="top", color=C["video"], fontsize=7, fontstyle="italic")
    ax.set_ylim(0, 62)
    ax.set_ylabel("Share of eligible video decisions (%)")
    ax.set_title("(a) ARC-SCEB chooses 4, 6, or 8 experts", loc="left", fontweight="bold")

    ax = axes[1]
    state_data = {
        "confident_audio": ([], []),
        "fragile_audio": ([], []),
        "insufficient_audio": ([], []),
    }
    for sample in cap["samples"]:
        for layer in sample["per_layer"]:
            hist = {int(k): int(v) for k, v in layer["chosen_m_hist"].items()}
            denom = sum(hist.values()) or 1
            kept = (hist.get(6, 0) + hist.get(8, 0)) / denom * 100
            xs, ys = state_data[layer["audio_state"]]
            xs.append(layer["audio_margin12_mean"] * 1e3)
            ys.append(kept)
    state_style = {
        "confident_audio": (C["text"], "o", "stable audio"),
        "fragile_audio": (C["guard"], "^", "uncertain audio"),
        "insufficient_audio": (C["m8"], "s", "too little audio"),
    }
    for state in state_style:
        xs, ys = state_data[state]
        if not xs:
            continue
        color, marker, label = state_style[state]
        ax.scatter(xs, ys, s=17, marker=marker, color=color, alpha=0.78,
                   edgecolor="white", linewidth=0.35,
                   label=f"{label} ({len(xs)})", zorder=3)
    audio_threshold = cap["policy"]["fragile_margin"] * 1e3
    ax.axvline(audio_threshold, color="black", linestyle="--", linewidth=0.9)
    ax.text(audio_threshold + 0.25, 103, r"$\tau_a$", ha="left", va="top", fontsize=7)
    fragile_y = state_data["fragile_audio"][1]
    n_states = sum(len(v[0]) for v in state_data.values())
    if fragile_y and min(fragile_y) >= 99.999:
        ax.text(0.98, 0.92, f"{len(fragile_y)}/{n_states} uncertain states\n"
                + r"all video budgets $\geq$6", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.7, color=C["guard"])
    ax.set_ylim(25, 105)
    ax.set_xlabel(r"Mean audio top-1/top-2 gap ($\times10^{-3}$)")
    ax.set_ylabel(r"Video decisions with $\geq$6 experts (%)")
    ax.set_title("(b) Uncertain audio keeps at least 6 experts", loc="left", fontweight="bold")
    ax.legend(loc="center right", handletextpad=0.25, borderaxespad=0.2)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.94, wspace=0.48)
    _save(fig, "policy_behavior")


# ============================================================= Figure 5
def fig5_schematic(agg, cap):
    """One-glance ARC-SCEB decision flow (design-motivation summary)."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, tc="black", fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                           linewidth=1.0, edgecolor="#444444", facecolor=fc, alpha=0.95)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, text="", color="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=11, lw=1.1, color=color))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, text, ha="center", fontsize=6.8,
                    color=color)

    box(0.1, 2.4, 1.5, 1.2, "Token-layer\nrouting probs\n(top-8)", "#EEEEEE", fs=7.5)
    # modality gate
    box(2.0, 4.2, 2.0, 1.0, "modality gate\ntext/audio/image/control\nremain top-8", C["text"], "white", 7.0)
    box(2.0, 2.6, 2.0, 1.0, "target video token\nin middle layers\n[12,32)?", C["video"], "white", 7.2)
    box(2.0, 0.6, 2.0, 1.0, "early/late layers\nremain top-8", "#DDDDDD", fs=7.2)
    arrow(1.6, 3.2, 2.0, 4.6)
    arrow(1.6, 3.0, 2.0, 3.1)
    arrow(1.6, 2.8, 2.0, 1.0)

    # boundary rule
    box(4.6, 2.5, 2.1, 1.2, "candidate gaps\n$p_4-p_5\\geq\\tau_b$ $\\rightarrow$ top-4\n$p_6-p_7\\geq\\tau_b$ $\\rightarrow$ top-6\nelse top-8", "#FFF3E0", fs=7)
    arrow(4.0, 3.1, 4.6, 3.1, "eligible")

    # audio floor
    box(4.6, 0.5, 2.1, 1.3, "one-way audio guard\ninsufficient $\\rightarrow$ floor 8\nfragile $\\rightarrow$ floor 6\nconfident $\\rightarrow$ unchanged", C["audio"], "white", 6.8)
    ax.add_patch(FancyArrowPatch((5.65, 2.5), (5.65, 1.8), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=C["guard"]))
    ax.text(6.05, 2.15, "same-sample\naudio router", ha="left", va="center", fontsize=6.6, color=C["guard"])

    # output
    box(7.3, 1.7, 1.6, 1.6, "dynamic mask\ntop-4 / 6 / 8\nzero tail weights\n+ renormalize", "#E3F2FD", fs=7.2)
    arrow(6.7, 3.0, 7.3, 2.8)
    arrow(6.7, 1.1, 7.3, 2.2)
    box(9.0, 2.1, 0.9, 0.9, "same MoE\ndispatch", "#EEEEEE", fs=6.6)
    arrow(8.9, 2.5, 9.0, 2.55)

    # top-m mix annotation from real capture
    if cap is not None:
        tot = {4: 0, 6: 0, 8: 0}
        for s in cap["samples"]:
            for k, v in s["chosen_m_hist_total"].items():
                tot[int(k)] += int(v)
        n = sum(tot.values()) or 1
        eff = (4 * tot[4] + 6 * tot[6] + 8 * tot[8]) / n
        ax.text(8.1, 1.35, f"mean {eff:.2f} experts\n/ video token", ha="center", fontsize=6.6,
                color=C["video"], style="italic")

    ax.set_title("ARC-SCEB dynamic policy: video-targeted, middle-layer, "
                 "boundary-budgeted, audio-guarded", fontsize=9, fontweight="bold", loc="left")
    _save(fig, "fig5_schematic")


def _save(fig, name):
    for ext in FORMATS:
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[fig] wrote {name}.{{{','.join(FORMATS)}}}")


# name -> (builder, needs_capture).  Edit here to add/rename figures.
FIGURES = {
    "paper_audit": (fig_paper_routing_audit, True),
    "paper_risk": (lambda agg, cap: fig_paper_risk_map(agg), False),
    "paper_modality": (lambda agg, cap: fig_paper_modality_target(agg), False),
    "paper_depth": (lambda agg, cap: fig_paper_depth_profile(agg), False),
    "paper_policy": (fig_paper_policy_behavior, True),
    "fig1": (lambda agg, cap: fig1_modality_dispersion(agg), False),
    "fig2": (lambda agg, cap: fig2_layer_stage(agg), False),
    "fig3": (lambda agg, cap: fig3_confidence_budget(agg, cap), True),
    "fig4": (lambda agg, cap: fig4_audio_control(agg, cap), True),
    "fig5": (lambda agg, cap: fig5_schematic(agg, cap), True),
}


def main():
    ap = argparse.ArgumentParser(
        description="Render ARC-SCEB interpretability figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), metavar="figN",
                    help="render only these figures (default: all)")
    ap.add_argument("--repo", default=None, help="repo root (default: auto-detected)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: <repo>/logs/arc_sceb_interpretability/figures)")
    ap.add_argument("--formats", nargs="+", default=None, metavar="EXT",
                    help="output formats, e.g. pdf png svg (default: pdf png)")
    args = ap.parse_args()

    _configure(repo=args.repo, outdir=args.outdir, formats=args.formats)
    agg, cap = load()
    print(f"[load] repo={REPO}")
    print(f"[load] agg benches={list(agg)} capture={'yes' if cap else 'no'}"
          + (f" ({cap['n_samples']} samples)" if cap else ""))

    todo = args.only or list(FIGURES)
    for name in todo:
        builder, needs_cap = FIGURES[name]
        if needs_cap and cap is None:
            print(f"[warn] {name} wants worker_capture.json but it is missing; "
                  f"drawing the capture-pending placeholder.")
        builder(agg, cap)
    print(f"[done] wrote {len(todo)} figure(s) ({','.join(FORMATS)}) to {FIG}")


if __name__ == "__main__":
    main()

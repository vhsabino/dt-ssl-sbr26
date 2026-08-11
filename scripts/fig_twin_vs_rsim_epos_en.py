#!/usr/bin/env python3
"""Regenerate the twin-vs-rSim fidelity figure in ENGLISH, e_pos panel only.

Usage: python3 scripts/fig_twin_vs_rsim_epos_en.py
Outputs: results/figs/twin_vs_rsim_epos_en_<timestamp>.{png,pdf}
         latex/figures/twin_vs_rsim_epos_en.png (paper candidate)
"""
import datetime
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Source: results/eval_twin_vs_rsim_modeA_20260617_0030.md, "trans-dominado"
# h (s), twin e_pos (m), rSim e_pos (m), gap (m), gap CI95 lo, gap CI95 hi
ROWS = [
    (0.10, 0.0077, 0.0572, 0.0495, 0.0451, 0.0524),
    (0.20, 0.0177, 0.0935, 0.0758, 0.0665, 0.0811),
    (0.40, 0.0382, 0.1312, 0.0930, 0.0790, 0.1029),
    (0.70, 0.0644, 0.1731, 0.1087, 0.0877, 0.1341),
    (1.00, 0.0919, 0.2144, 0.1225, 0.1045, 0.1452),
    (1.50, 0.1229, 0.2927, 0.1699, 0.1188, 0.2101),
]
N_TOTAL, N_INDEP, H_REF = 264, 49, 0.40

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

h = [r[0] for r in ROWS]
twin = [r[1] for r in ROWS]
rsim = [r[2] for r in ROWS]
gap = [r[3] for r in ROWS]
lo = [r[3] - r[4] for r in ROWS]
hi = [r[5] - r[3] for r in ROWS]

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(3.4, 1.85), dpi=400, gridspec_kw={"wspace": 0.42}
)

# Left: median position error vs horizon (print-safe: solid vs dashed, markers)
ax.plot(h, twin, "-o", color="black", lw=1.2, ms=3.2, label="twin (FOPDT)")
ax.plot(h, rsim, "--s", color="0.45", lw=1.2, ms=3.2, label="rSim (ODE)")
ax.axvline(H_REF, color="0.75", lw=0.7, ls=":", zorder=0)
ax.set_xlabel("horizon $h$ (s)", fontsize=6)
ax.set_ylabel("median $e_{\\mathrm{pos}}$ (m)", fontsize=6)
ax.tick_params(labelsize=5.5, length=2)
ax.legend(fontsize=5, frameon=False, loc="upper left")
ax.grid(alpha=0.25, lw=0.4)

# Right: gap with moving-block bootstrap 95% CI
ax2.errorbar(
    h, gap, yerr=[lo, hi], fmt="-o", color="black", lw=1.2, ms=3.2,
    capsize=2, elinewidth=0.8,
)
ax2.axhline(0.0, color="0.6", lw=0.7)
ax2.axvline(H_REF, color="0.75", lw=0.7, ls=":", zorder=0)
ax2.set_xlabel("horizon $h$ (s)", fontsize=6)
ax2.set_ylabel("gap (m), rSim $-$ twin", fontsize=6)
ax2.tick_params(labelsize=5.5, length=2)
ax2.grid(alpha=0.25, lw=0.4)

for a in (ax, ax2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

figs = ROOT / "results" / "figs"
figs.mkdir(parents=True, exist_ok=True)
out_png = figs / f"twin_vs_rsim_epos_en_{STAMP}.png"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(figs / f"twin_vs_rsim_epos_en_{STAMP}.pdf", bbox_inches="tight")

cand = ROOT / "latex" / "figures" / "twin_vs_rsim_epos_en.png"
cand.write_bytes(out_png.read_bytes())
print(f"wrote {out_png}")
print(f"wrote {cand}")
print(f"N_total={N_TOTAL} N_indep={N_INDEP} h_ref={H_REF}")

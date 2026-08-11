"""Relatorio HONESTO de TODAS as janelas do hold-out (sem cherry-pick):
distribuicao de bins, tabela completa de erros por janela, distribuicao do gap, e
overlays REPRESENTATIVOS (mediano/melhor/pior caso do twin) + contact-sheet.

RENDERIZADOR/RELATOR FINO: NAO recomputa dinamica nem o gate, NAO toca em banco.
Reusa o harness existente:
  - splits.load_holdout_segments            (mesma fonte de hold-out)
  - rollout.rollout_segment                 (mesmo motor, H_max=90 = 1.5 s)
  - plant.AnalyticFOPDTPlant / rsim_plant.RSimReplay(warmup=True)  (modo A do gate)
  - metrics.build_error_table / bin_distribution / classify_bin
        (a MESMA classificacao de bin e a MESMA flag de independencia do gate)
  - eval_twin_vs_rsim._silence_engine_stdout

A populacao aqui e o HOLD-OUT (10 eventos), NAO as 5 partidas: PREVIA, nao a
afirmacao final sobre operacao.

Uso:
  ~/miniforge3/envs/rsim310/bin/python -m phase0_replay.holdout_report
"""
from __future__ import annotations

import csv
from datetime import datetime

import numpy as np

from .io import repo_root
from .splits import load_holdout_segments
from .plant import AnalyticFOPDTPlant
from .rsim_plant import RSimReplay
from .rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE, DEFAULT_HORIZONS
from .metrics import (build_error_table, bin_distribution, load_metrics_config,
                      BIN_LABEL, BIN_ORDER)
from .params import load_twin_params
from .eval_twin_vs_rsim import _silence_engine_stdout

LW = 15           # warm-up (modo A do gate)
SUB_MS = 1
H_1S = 60         # 1.0 s @ 1/60
H_15 = 90         # 1.5 s @ 1/60  (= H_max)


# --------------------------------------------------------------------------
def _build():
    """Constroi rollouts pareados (modo A), a ErrorTable canonica e o mapa
    linha->(segmento, janela) para recuperar trajetorias."""
    params = load_twin_params()
    mcfg = load_metrics_config()
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)

    twin_ros = [rollout_segment(s, AnalyticFOPDTPlant(params), h_max=DEFAULT_HMAX,
                                stride=DEFAULT_STRIDE, lookback=LW,
                                horizons=DEFAULT_HORIZONS) for s in segs]
    with _silence_engine_stdout():
        rsim_ros = [rollout_segment(s, RSimReplay(warmup=True, sub_ms=SUB_MS),
                                    h_max=DEFAULT_HMAX, stride=DEFAULT_STRIDE,
                                    lookback=LW, horizons=DEFAULT_HORIZONS)
                    for s in segs]

    table = build_error_table(twin_ros, rsim_ros, params=params, mcfg=mcfg)
    # mapa linha->(si,wi): MESMA ordem de iteracao do build_error_table (sem skip)
    row_to_win = [(si, wi) for si, rt in enumerate(twin_ros)
                  for wi in range(len(rt.windows))]
    assert len(row_to_win) == table.W, "mapa de linhas dessincronizado com a tabela"
    return params, mcfg, segs, dropped, twin_ros, rsim_ros, table, row_to_win


def _hidx(table, h):
    return int(np.flatnonzero(table.horizons == h)[0])


def _final_err_cm(pose_sim, pose_real):
    d = pose_sim[-1, :2] - pose_real[-1, :2]
    return float(np.hypot(d[0], d[1]) * 100.0)


# --------------------------------------------------------------------------
# (2) DISTRIBUICAO DE BINS
def _bin_distribution(table, outdir, figdir, stamp):
    d = bin_distribution(table)
    Wtot = d["all"]["total"]
    rows = []
    for b in BIN_ORDER:
        rows.append((BIN_LABEL[b], d[b]["total"], d[b]["indep"],
                     d[b]["total"] / Wtot if Wtot else 0.0))
    rows.append((BIN_LABEL.get("all", "TODOS"), d["all"]["total"], d["all"]["indep"], 1.0))

    csvp = outdir / f"bin_distribution_{stamp}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "janelas_densas", "janelas_independentes", "fracao_densas"])
        for name, tot, ind, frac in rows:
            w.writerow([name, tot, ind, f"{frac:.4f}"])

    mdp = outdir / f"bin_distribution_{stamp}.md"
    L = ["# Distribuicao de bins — HOLD-OUT (10 eventos) — PREVIA, nao a operacao",
         "",
         "Populacao = janelas do **hold-out (10 eventos)**, NAO as 5 partidas. "
         "Resultado de previa; nao e a afirmacao final sobre operacao.",
         "",
         f"Classificacao (metrics.py): razao = ativ_rot/(ativ_rot+ativ_trans); "
         f"trans<0.2 / misto 0.2-0.5 / rot>0.5. H_max=90 (1.5 s), modo A (Lw={LW}).",
         "",
         "| bin | janelas densas | janelas independentes | fracao (densas) |",
         "|---|---:|---:|---:|"]
    for name, tot, ind, frac in rows:
        bold = "**" if name.startswith("TODOS") else ""
        L.append(f"| {bold}{name}{bold} | {tot} | {ind} | {frac:.3f} |")
    mdp.write_text("\n".join(L) + "\n", encoding="utf-8")

    # grafico de barras da fracao por bin
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = [BIN_LABEL[b] for b in BIN_ORDER]
    fracs = [d[b]["total"] / Wtot if Wtot else 0.0 for b in BIN_ORDER]
    colors = [(0.20, 0.45, 0.70), (0.85, 0.65, 0.13), (0.70, 0.20, 0.20)]
    bars = ax.bar(labels, fracs, color=colors, edgecolor="0.3")
    for bar, fr, b in zip(bars, fracs, BIN_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2, fr + 0.01,
                f"{fr*100:.1f}%\n(n={d[b]['total']})", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel("fracao das janelas densas")
    ax.set_ylim(0, max(fracs) * 1.25 if fracs else 1)
    ax.set_title(f"Distribuicao de bins — HOLD-OUT (10 eventos), N={Wtot} janelas\n"
                 "(PREVIA — nao as 5 partidas / nao a operacao)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    base = figdir / f"bins_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return csvp, mdp, f"{base}.png", f"{base}.svg", d, rows


# --------------------------------------------------------------------------
# (3) TABELA COMPLETA POR JANELA
def _window_table(table, twin_ros, rsim_ros, row_to_win, outdir, stamp):
    i1, i15 = _hidx(table, H_1S), _hidx(table, H_15)
    ept, epd = table.epos_twin, table.epos_def
    csvp = outdir / f"window_errors_{stamp}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segmento", "t0_idx", "t0_s", "bin", "indep",
                    "e_pos_twin_1.0s_m", "e_pos_grsim_1.0s_m", "gap_1.0s_m",
                    "e_pos_twin_1.5s_m", "e_pos_grsim_1.5s_m", "gap_1.5s_m"])
        for r in range(table.W):
            si, wi = row_to_win[r]
            win = twin_ros[si].windows[wi]
            g1 = epd[r, i1] - ept[r, i1]
            g15 = epd[r, i15] - ept[r, i15]
            w.writerow([table.seg_name[r], int(table.t0[r]), f"{win.t0_time:.4f}",
                        table.bin[r], int(table.indep[r]),
                        f"{ept[r,i1]:.4f}", f"{epd[r,i1]:.4f}", f"{g1:+.4f}",
                        f"{ept[r,i15]:.4f}", f"{epd[r,i15]:.4f}", f"{g15:+.4f}"])
    return csvp, i1, i15


# --------------------------------------------------------------------------
# (4) DISTRIBUICAO DO GAP (bin de translacao, h=1.0s)
def _gap_distribution(table, i1, figdir, stamp):
    trans = table.bin == "trans"
    gap = (table.epos_def[trans, i1] - table.epos_twin[trans, i1])  # m
    gap_cm = gap * 100.0
    med = float(np.median(gap_cm))
    frac_twin_win = float(np.mean(gap > 0))
    n_grsim_win = int(np.sum(gap < 0))
    n = gap.size

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(gap_cm, bins=40, color=(0.30, 0.50, 0.75), edgecolor="0.3", alpha=0.85)
    ax.axvline(0, color="black", lw=1.5, ls="--", label="empate (gap=0)")
    ax.axvline(med, color=(0.75, 0.20, 0.15), lw=2.0,
               label=f"mediana = {med:+.1f} cm")
    ax.set_xlabel("gap = e_pos(grSim) - e_pos(twin)  [cm]   (>0 => twin melhor)")
    ax.set_ylabel("nº de janelas")
    ax.set_title("Distribuicao do gap por janela @ h=1.0s — bin TRANSLACAO\n"
                 "HOLD-OUT (10 eventos) — PREVIA", fontsize=11)
    ax.grid(alpha=0.3)
    txt = (f"N = {n} janelas trans\n"
           f"twin < grSim: {frac_twin_win*100:.1f}%  ({n - n_grsim_win}/{n})\n"
           f"grSim vence: {n_grsim_win} janela(s)\n"
           f"mediana gap = {med:+.1f} cm")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, va="top", ha="right",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    base = figdir / f"gap_dist_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return f"{base}.png", f"{base}.svg", med, frac_twin_win, n_grsim_win, n


# --------------------------------------------------------------------------
# overlay helper
def _draw_overlay(ax, win_t, win_d, *, full=True):
    pr, pt, ps = win_t.pose_real, win_t.pose_sim, win_d.pose_sim
    ax.plot(pr[:, 0], pr[:, 1], "-", color="black", lw=(3.0 if full else 1.2), zorder=3)
    ax.plot(pt[:, 0], pt[:, 1], "-", color=(0.10, 0.30, 0.65),
            lw=(2.0 if full else 0.9), zorder=4)
    ax.plot(ps[:, 0], ps[:, 1], "-", color=(0.75, 0.20, 0.15),
            lw=(2.0 if full else 0.9), zorder=4)
    ax.plot(pr[0, 0], pr[0, 1], "o", color="black", ms=(9 if full else 3),
            mfc="white", mew=(1.8 if full else 0.8), zorder=5)
    ax.set_aspect("equal", adjustable="datalim")


# --------------------------------------------------------------------------
# (5) OVERLAYS REPRESENTATIVOS (mediano / melhor / pior caso do twin)
def _representative(table, twin_ros, rsim_ros, row_to_win, i1, i15, figdir, stamp):
    trans_rows = np.flatnonzero(table.bin == "trans")
    gap = table.epos_def[trans_rows, i1] - table.epos_twin[trans_rows, i1]
    order = np.argsort(gap)
    r_min = trans_rows[order[0]]                       # menor gap = PIOR caso twin
    r_max = trans_rows[order[-1]]                      # maior gap = melhor caso twin
    med_val = np.median(gap)
    r_med = trans_rows[order[int(np.argmin(np.abs(np.sort(gap) - med_val)))]]
    # argmin sobre gap ordenado -> reindexa p/ a linha original
    k_med = int(np.argmin(np.abs(gap - med_val)))
    r_med = trans_rows[k_med]

    picks = [("caso mediano", r_med), ("melhor caso (twin)", r_max),
             ("PIOR caso (twin)", r_min)]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    chosen = []
    for ax, (label, r) in zip(axes, picks):
        si, wi = row_to_win[r]
        wt, wd = twin_ros[si].windows[wi], rsim_ros[si].windows[wi]
        _draw_overlay(ax, wt, wd, full=True)
        et = _final_err_cm(wt.pose_sim, wt.pose_real)
        es = _final_err_cm(wd.pose_sim, wt.pose_real)
        gap_cm = (table.epos_def[r, i1] - table.epos_twin[r, i1]) * 100.0
        ax.grid(alpha=0.3, lw=0.6)
        ax.set_xlabel("x campo (m)"); ax.set_ylabel("y campo (m)")
        ax.set_title(f"{label}\n{table.seg_name[r]} (t0={wt.t0_time:.2f}s, "
                     f"gap@1.0s={gap_cm:+.0f} cm)", fontsize=10)
        ax.text(0.03, 0.97, f"erro final @1.5s:\ntwin = {et:.1f} cm\n"
                f"grSim = {es:.1f} cm", transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
        chosen.append((label, table.seg_name[r], float(wt.t0_time), float(gap_cm), et, es))
    # legenda unica
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", lw=3, label="real (ground truth)"),
               Line2D([0], [0], color=(0.10, 0.30, 0.65), lw=2, label="twin (FOPDT)"),
               Line2D([0], [0], color=(0.75, 0.20, 0.15), lw=2, label="grSim/rSim (ODE)"),
               Line2D([0], [0], color="black", marker="o", mfc="white", ls="",
                      label="inicio (t0)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Overlays representativos (bin translacao, ranqueados por gap@1.0s) "
                 "— HOLD-OUT, sem cherry-pick", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    base = figdir / f"representative_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return f"{base}.png", f"{base}.svg", chosen


# --------------------------------------------------------------------------
# (6) CONTACT-SHEET de TODAS as janelas de translacao
def _contact_sheet(table, twin_ros, rsim_ros, row_to_win, figdir, stamp):
    trans_rows = np.flatnonzero(table.bin == "trans")
    n = trans_rows.size
    ncols = 14
    nrows = int(np.ceil(n / ncols))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.25, nrows * 1.25))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, trans_rows):
        si, wi = row_to_win[r]
        _draw_overlay(ax, twin_ros[si].windows[wi], rsim_ros[si].windows[wi], full=False)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Contact-sheet — TODAS as {n} janelas de translacao (HOLD-OUT) "
                 "| preto=real, azul=twin, vermelho=grSim", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    base = figdir / f"contact_trans_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return f"{base}.png", f"{base}.svg", n


# --------------------------------------------------------------------------
def run():
    import matplotlib
    matplotlib.use("Agg")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resdir = repo_root() / "results"
    figdir = resdir / "figs"
    resdir.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)

    (params, mcfg, segs, dropped, twin_ros, rsim_ros,
     table, row_to_win) = _build()

    bin_csv, bin_md, bins_png, bins_svg, d, bin_rows = _bin_distribution(
        table, resdir, figdir, stamp)
    win_csv, i1, i15 = _window_table(table, twin_ros, rsim_ros, row_to_win, resdir, stamp)
    gap_png, gap_svg, gap_med, frac_twin, n_grsim, n_trans = _gap_distribution(
        table, i1, figdir, stamp)
    rep_png, rep_svg, chosen = _representative(
        table, twin_ros, rsim_ros, row_to_win, i1, i15, figdir, stamp)
    cs_png, cs_svg, n_cs = _contact_sheet(
        table, twin_ros, rsim_ros, row_to_win, figdir, stamp)

    # ---- relatorio em console ----
    print("=" * 74)
    print("RELATORIO HONESTO — TODAS as janelas do HOLD-OUT (PREVIA, nao a operacao)")
    print(f"segmentos={len(segs)} (idle dropped={len(dropped)}); janelas totais={table.W}; "
          f"H_max=90 (1.5s); modo A (Lw={LW}, sub_ms={SUB_MS})")
    print("=" * 74)
    print("\n[2] DISTRIBUICAO DE BINS (HOLD-OUT 10 eventos):")
    print(f"  {'bin':16s} {'densas':>7s} {'indep':>7s} {'fracao':>8s}")
    for name, tot, ind, frac in bin_rows:
        print(f"  {name:16s} {tot:7d} {ind:7d} {frac:8.3f}")
    print("\n[4] GAP (bin translacao, h=1.0s):")
    print(f"  N trans={n_trans} | mediana gap={gap_med:+.1f} cm | "
          f"twin<grSim={frac_twin*100:.1f}% | grSim vence={n_grsim} janela(s)")
    print("\n[5] OVERLAYS REPRESENTATIVOS (t0 escolhidos):")
    for label, seg, t0s, gcm, et, es in chosen:
        print(f"  {label:20s}: {seg} @ t0={t0s:.2f}s | gap@1.0s={gcm:+.0f} cm | "
              f"twin_final={et:.1f}cm grSim_final={es:.1f}cm")

    print("\n--- ARQUIVOS ---")
    for p in (bin_csv, bin_md, win_csv):
        print(f"  {p}")
    for p in (bins_png, bins_svg, gap_png, gap_svg, rep_png, rep_svg, cs_png, cs_svg):
        print(f"  {p}")
    return dict(stamp=stamp, table=table, chosen=chosen)


if __name__ == "__main__":
    run()

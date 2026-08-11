"""Overlays de PERCURSO COMPLETO (free-run open-loop do segmento INTEIRO).

VISUALIZACAO, NAO metrica: NAO recomputa GO; o erro final NAO e metrica validada
(o numero validado e o de 1.5 s, marcado com 'x'). Alem de 1.5 s e extrapolacao
open-loop => divergencia cumulativa esperada.

RENDERIZADOR FINO: NAO cria infra, NAO toca em banco. Reusa o harness:
  - splits.load_holdout_segments
  - rollout.free_run / max_command_delay      (mesmo motor, mesmo priming/IC)
  - plant.AnalyticFOPDTPlant / rsim_plant.RSimReplay(warmup=True)  (modo A do gate)
  - eval_twin_vs_rsim._silence_engine_stdout

Diferenca p/ o rollout do gate: UMA unica janela = segmento inteiro (sem janelar
em 1.5 s). reset em idx_min (apos priming de atraso) com a pose e a vel-de-corpo
REAIS desse indice; prime_delay com os comandos anteriores; free-run ate o fim.

Uso:
  ~/miniforge3/envs/rsim310/bin/python -m phase0_replay.percurso_completo
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .io import repo_root
from .splits import load_holdout_segments
from .plant import AnalyticFOPDTPlant
from .rsim_plant import RSimReplay
from .rollout import free_run, max_command_delay
from .params import load_twin_params
from .eval_twin_vs_rsim import _silence_engine_stdout

LW = 15            # warm-up / lookback (modo A do gate)
SUB_MS = 1
H_VALID = 90       # 1.5 s @ 1/60 — fronteira da metrica validada
W_HI = 1.5         # rad/s — limiar de |w_cmd| p/ destacar trechos de ROTACAO

C_REAL = (0.0, 0.0, 0.0)
C_TWIN = (0.10, 0.30, 0.65)
C_RSIM = (0.75, 0.20, 0.15)
C_ROT = (0.95, 0.55, 0.10)


def _full_freerun(seg, plant, params):
    """UM free-run cobrindo o segmento inteiro. Retorna (pose_sim, pose_real, t_rel,
    w_cmd_path, idx_min). Espelha a montagem de janela do rollout, mas com a janela
    = segmento todo."""
    dt = seg.ts
    N = seg.n
    md = max_command_delay(dt, params)
    idx_min = md + 1
    cmd = np.column_stack([seg.cmd_vx, seg.cmd_vy, seg.cmd_w])
    velb = np.column_stack([seg.velb_x, seg.velb_y, seg.velb_w])
    pos = np.column_stack([seg.pos_x, seg.pos_y, seg.theta])

    L = N - 1 - idx_min                      # passos do free-run completo
    pose0 = (float(pos[idx_min, 0]), float(pos[idx_min, 1]), float(pos[idx_min, 2]))
    velb0 = (float(velb[idx_min, 0]), float(velb[idx_min, 1]), float(velb[idx_min, 2]))
    P = min(idx_min, LW)
    prev = cmd[idx_min - P:idx_min]
    win_cmds = cmd[idx_min:idx_min + L]
    pose_sim = free_run(plant, win_cmds, pose0, velb0, dt, prev_cmds=prev)
    pose_real = pos[idx_min:idx_min + L + 1].copy()
    w_cmd_path = np.abs(seg.cmd_w[idx_min:idx_min + L + 1])   # alinhado ao path
    t_rel = np.arange(L + 1) * dt
    return pose_sim, pose_real, t_rel, w_cmd_path, idx_min


def _epos_cm(a, b):
    d = a[:, :2] - b[:, :2]
    return np.hypot(d[:, 0], d[:, 1]) * 100.0


def _tortuosity(pose_real):
    """pathlen/netdisp da trajetoria real — heuristica de 'forma fechada/complexa'."""
    xy = pose_real[:, :2]
    seglen = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()
    net = np.hypot(*(xy[-1] - xy[0]))
    return float(seglen / net) if net > 1e-6 else float("inf"), float(seglen), float(net)


def _draw_xy(ax, R, T, S, w_path, *, full=True, with_legend=False):
    ax.plot(R[:, 0], R[:, 1], "-", color=C_REAL, lw=(3.0 if full else 1.4),
            zorder=3, label="real (ground truth)")
    ax.plot(T[:, 0], T[:, 1], "-", color=C_TWIN, lw=(2.0 if full else 1.0),
            zorder=4, label="twin (FOPDT)")
    ax.plot(S[:, 0], S[:, 1], "-", color=C_RSIM, lw=(2.0 if full else 1.0),
            zorder=4, label="grSim/rSim (ODE)")
    # ponto inicial (comum)
    ax.plot(R[0, 0], R[0, 1], "o", color="black", ms=(9 if full else 4),
            mfc="white", mew=1.6, zorder=6, label="inicio (t0)")
    # marcador 'x' em t0+1.5 s (fronteira da metrica validada)
    if R.shape[0] > H_VALID:
        ms = 11 if full else 6
        ax.plot(R[H_VALID, 0], R[H_VALID, 1], "x", color=C_REAL, ms=ms, mew=2.2, zorder=7,
                label="t0+1.5 s (metrica validada)")
        ax.plot(T[H_VALID, 0], T[H_VALID, 1], "x", color=C_TWIN, ms=ms, mew=2.2, zorder=7)
        ax.plot(S[H_VALID, 0], S[H_VALID, 1], "x", color=C_RSIM, ms=ms, mew=2.2, zorder=7)
    # destaque de ROTACAO na trajetoria REAL (|w_cmd| alto)
    hi = w_path >= W_HI
    if hi.any():
        ax.scatter(R[hi, 0], R[hi, 1], s=(28 if full else 8), marker="^",
                   color=C_ROT, edgecolors="0.25", linewidths=0.4, zorder=5,
                   label=f"real: |w_cmd|>={W_HI} rad/s")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3, lw=0.6)
    if with_legend:
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85)


def _per_segment_fig(name, R, T, S, t_rel, w_path, figdir, stamp, extra_label=None):
    import matplotlib.pyplot as plt
    dur = float(t_rel[-1])
    et = _epos_cm(T, R); es = _epos_cm(S, R)
    et_f, es_f = float(et[-1]), float(es[-1])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 10.5),
                                   gridspec_kw=dict(height_ratios=[2.4, 1.0]))
    _draw_xy(ax0, R, T, S, w_path, full=True, with_legend=True)
    ax0.set_xlabel("x campo (m)"); ax0.set_ylabel("y campo (m)")
    banner = f"  [{extra_label}]" if extra_label else ""
    ax0.set_title(f"{name}{banner}  —  dur={dur:.1f}s\n"
                  "free-run completo open-loop — divergencia cumulativa; "
                  "metrica validada = 1.5 s (marcador x)", fontsize=10)
    ax0.text(0.03, 0.97, f"erro final (NAO metrica):\ntwin = {et_f:.1f} cm\n"
             f"grSim = {es_f:.1f} cm", transform=ax0.transAxes, va="top", ha="left",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    # companheiro: erro de posicao vs tempo
    ax1.plot(t_rel, et, "-", color=C_TWIN, lw=1.8, label="twin")
    ax1.plot(t_rel, es, "-", color=C_RSIM, lw=1.8, label="grSim")
    ax1.axvline(1.5, color="black", ls="--", lw=1.4, label="1.5 s (metrica validada)")
    ax1.axvspan(1.5, dur, color="0.85", alpha=0.4, zorder=0)
    ax1.text(0.5 * (1.5 + dur), ax1.get_ylim()[1] * 0.92, "extrapolacao open-loop",
             ha="center", va="top", fontsize=8, color="0.35")
    ax1.set_xlabel("t desde t0 (s)"); ax1.set_ylabel("e_pos (cm)")
    ax1.set_xlim(0, dur); ax1.grid(alpha=0.3); ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    base = figdir / f"percurso_completo_{name}_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return f"{base}.png", dur, et_f, es_f


def _grid_fig(items, figdir, stamp, *, suptitle=None, fname="percurso_completo_grid"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(items)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, items):
        _draw_xy(ax, it["R"], it["T"], it["S"], it["w"], full=False)
        ratio = it["es_f"] / it["et_f"] if it["et_f"] > 1e-6 else float("inf")
        ax.set_title(f"{it['name']}\ndur={it['dur']:.1f}s  grSim/twin={ratio:.1f}x",
                     fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=C_REAL, lw=3, label="real (ground truth)"),
               Line2D([0], [0], color=C_TWIN, lw=2, label="twin (FOPDT)"),
               Line2D([0], [0], color=C_RSIM, lw=2, label="grSim/rSim (ODE)"),
               Line2D([0], [0], color=C_REAL, marker="x", ls="", label="t0+1.5 s (validado)"),
               Line2D([0], [0], color=C_ROT, marker="^", ls="", label=f"|w_cmd|>={W_HI} rad/s"),
               Line2D([0], [0], color="black", marker="o", mfc="white", ls="", label="inicio")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(suptitle or
                 "Percurso completo (free-run open-loop) — TODOS os segmentos hold-out\n"
                 "VISUALIZACAO, nao metrica; alem do 'x' (1.5 s) e extrapolacao",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    base = figdir / f"{fname}_{stamp}"
    fig.savefig(f"{base}.png", dpi=200); fig.savefig(f"{base}.svg")
    plt.close(fig)
    return f"{base}.png", f"{base}.svg"


def run():
    import matplotlib
    matplotlib.use("Agg")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figdir = repo_root() / "results" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    params = load_twin_params()
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)

    # free-runs: twin sem silenciar; grSim silenciando o engine C++.
    twin_runs = [_full_freerun(s, AnalyticFOPDTPlant(params), params) for s in segs]
    with _silence_engine_stdout():
        rsim_runs = [_full_freerun(s, RSimReplay(warmup=True, sub_ms=SUB_MS), params)
                     for s in segs]

    items, per_files = [], []
    for s, tw, rs in zip(segs, twin_runs, rsim_runs):
        name = str(s.meta.get("event"))
        T, R, t_rel, w_path, _ = tw
        S = rs[0]                              # pose_sim do grSim (mesmas R/t_rel)
        png, dur, et_f, es_f = _per_segment_fig(name, R, T, S, t_rel, w_path, figdir, stamp)
        per_files.append(png)
        tort, plen, net = _tortuosity(R)
        items.append(dict(name=name, R=R, T=T, S=S, w=w_path, dur=dur,
                          et_f=et_f, es_f=es_f, tort=tort, plen=plen, net=net))

    grid_png, grid_svg = _grid_fig(items, figdir, stamp)

    # ---- relatorio ----
    print("=" * 86)
    print("PERCURSO COMPLETO (free-run open-loop) — VISUALIZACAO, nao metrica")
    print("erro final NAO e metrica do modelo; o numero validado e o de 1.5 s (marcador x).")
    print(f"segmentos={len(segs)} (idle dropped={len(dropped)}); modo A (Lw={LW}); "
          f"limiar rotacao |w_cmd|>={W_HI} rad/s")
    print("=" * 86)
    print(f"{'segmento':28} {'dur(s)':>7} {'twin_fin(cm)':>12} {'grSim_fin(cm)':>13} "
          f"{'grSim/twin':>11} {'tortuos.':>9}")
    for it in sorted(items, key=lambda x: x["tort"], reverse=True):
        ratio = it["es_f"] / it["et_f"] if it["et_f"] > 1e-6 else float("inf")
        print(f"{it['name'][:28]:28} {it['dur']:>7.1f} {it['et_f']:>12.1f} "
              f"{it['es_f']:>13.1f} {ratio:>10.1f}x {it['tort']:>9.2f}")

    top = sorted(items, key=lambda x: x["tort"], reverse=True)[:3]
    print("\nFORMA MAIS FECHADA/COMPLEXA (maior tortuosidade = mais vai-e-vem/voltas):")
    for it in top:
        print(f"  {it['name']}: tortuosidade={it['tort']:.2f} "
              f"(pathlen={it['plen']:.2f} m, desloc_liq={it['net']:.2f} m) "
              f"-> candidato a slide")

    print("\n--- ARQUIVOS ---")
    for p in per_files:
        print(f"  {p}")
    print(f"  {grid_png}")
    print(f"  {grid_svg}")
    return items, stamp


if __name__ == "__main__":
    run()

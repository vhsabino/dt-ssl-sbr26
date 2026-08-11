"""Figura de OVERLAY DE TRAJETORIA (frame de CAMPO) para reuniao.

RENDERIZADOR FINO: so desenha as trajetorias que o rollout JA produz
(real x twin FOPDT x rSim/grSim ODE). NAO recomputa o gate, NAO mexe em
metricas, NAO toca em banco. Reusa INTEGRALMENTE o harness existente:
  - splits.load_holdout_segments       (mesma fonte de hold-out)
  - rollout.rollout_segment / free_run (mesmo motor, H_max=90 = 1.5 s)
  - plant.AnalyticFOPDTPlant           (twin)
  - rsim_plant.RSimReplay(warmup=True) (rSim, MESMA config do modo A do gate)
  - metrics.classify_bin/_window_activity (a MESMA classificacao de bin)
  - eval_twin_vs_rsim._silence_engine_stdout (silencia o engine C++)

Selecao: 3 segmentos hold-out com janelas TRANSLACAO-DOMINADAS, os de maior
deslocamento (clareza visual); 1-2 janelas por segmento -> grid 2x3.

Uso:
  ~/miniforge3/envs/rsim310/bin/python -m phase0_replay.meeting_overlay
"""
from __future__ import annotations

import contextlib
from datetime import datetime

import numpy as np

from .io import repo_root
from .splits import load_holdout_segments
from .plant import AnalyticFOPDTPlant
from .rsim_plant import RSimReplay
from .rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE, DEFAULT_HORIZONS
from .metrics import (load_metrics_config, classify_bin, _window_activity,
                      _get_robot_radius, BIN_LABEL)
from .params import load_twin_params
from .eval_twin_vs_rsim import _silence_engine_stdout

LW = 15           # warm-up (modo A do gate)
SUB_MS = 1
N_SEG = 3         # segmentos selecionados
WIN_PER_SEG = 2   # janelas por segmento -> 3x2 = 6 paineis (grid 2x3)


def _final_pos_err_cm(pose_sim, pose_real) -> float:
    """Erro de posicao FINAL (h = H_max) em cm, no frame de campo."""
    d = pose_sim[-1, :2] - pose_real[-1, :2]
    return float(np.hypot(d[0], d[1]) * 100.0)


def _net_disp(pose_real) -> float:
    """Deslocamento liquido real na janela (m), p/ ranquear clareza visual."""
    d = pose_real[-1, :2] - pose_real[0, :2]
    return float(np.hypot(d[0], d[1]))


def _collect(h_max=DEFAULT_HMAX, stride=DEFAULT_STRIDE):
    params = load_twin_params()
    mcfg = load_metrics_config()
    robot_radius = _get_robot_radius()
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)

    # rollouts pareados (mesmas janelas/t0), twin e rSim — modo A (warm-up).
    twin_ros, rsim_ros = [], []
    for s in segs:
        twin_ros.append(rollout_segment(s, AnalyticFOPDTPlant(params), h_max=h_max,
                                        stride=stride, lookback=LW,
                                        horizons=DEFAULT_HORIZONS))
    with _silence_engine_stdout():
        for s in segs:
            rsim_ros.append(rollout_segment(s, RSimReplay(warmup=True, sub_ms=SUB_MS),
                                            h_max=h_max, stride=stride, lookback=LW,
                                            horizons=DEFAULT_HORIZONS))

    ts = twin_ros[0].meta["ts"]
    # candidatos: SO janelas trans-dominadas (mesma classificacao do metrics.py)
    per_seg: dict[int, list] = {}
    for si, (rt, rd) in enumerate(zip(twin_ros, rsim_ros)):
        for wt, wd in zip(rt.windows, rd.windows):
            if wt.idle:
                continue
            _, _, razao = _window_activity(wt, robot_radius, ts)
            if classify_bin(razao, mcfg["bins"]) != "trans":
                continue
            per_seg.setdefault(si, []).append(dict(
                seg_idx=si, event=rt.meta.get("event"), t0=wt.t0,
                t0_time=wt.t0_time, disp=_net_disp(wt.pose_real),
                pose_real=wt.pose_real, pose_twin=wt.pose_sim, pose_rsim=wd.pose_sim,
                err_twin=_final_pos_err_cm(wt.pose_sim, wt.pose_real),
                err_rsim=_final_pos_err_cm(wd.pose_sim, wt.pose_real)))

    # 3 segmentos com a MAIOR janela trans (deslocamento), p/ clareza visual.
    seg_rank = sorted(per_seg, key=lambda s: max(w["disp"] for w in per_seg[s]),
                      reverse=True)[:N_SEG]

    panels = []
    for si in seg_rank:
        wins = sorted(per_seg[si], key=lambda w: w["disp"], reverse=True)
        chosen, picked_t0 = [], []
        for w in wins:                       # prioriza janelas NAO sobrepostas
            if all(abs(w["t0"] - pt) >= h_max for pt in picked_t0):
                chosen.append(w); picked_t0.append(w["t0"])
            if len(chosen) >= WIN_PER_SEG:
                break
        for w in wins:                       # completa se faltou (segmento curto)
            if len(chosen) >= WIN_PER_SEG:
                break
            if w not in chosen:
                chosen.append(w)
        panels.extend(chosen)
    return panels, ts, h_max, len(segs), len(dropped)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, ts, h_max, n_seg, n_drop = _collect()
    horizon_s = h_max * ts

    C_REAL = (0.0, 0.0, 0.0)
    C_TWIN = (0.10, 0.30, 0.65)
    C_RSIM = (0.75, 0.20, 0.15)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9.5))
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, panels):
        xr, yr = p["pose_real"][:, 0], p["pose_real"][:, 1]
        xt, yt = p["pose_twin"][:, 0], p["pose_twin"][:, 1]
        xs, ys = p["pose_rsim"][:, 0], p["pose_rsim"][:, 1]
        ax.plot(xr, yr, "-", color=C_REAL, lw=3.0, label="real (ground truth)", zorder=3)
        ax.plot(xt, yt, "-", color=C_TWIN, lw=2.0, label="twin (FOPDT)", zorder=4)
        ax.plot(xs, ys, "-", color=C_RSIM, lw=2.0, label="grSim/rSim (ODE)", zorder=4)
        # ponto inicial (comum aos tres em t0)
        ax.plot(xr[0], yr[0], "o", color="black", ms=9, mfc="white", mew=1.8,
                zorder=5, label="inicio (t0)")
        # pontos finais
        ax.plot(xr[-1], yr[-1], "s", color=C_REAL, ms=6, zorder=5)
        ax.plot(xt[-1], yt[-1], "d", color=C_TWIN, ms=6, zorder=5)
        ax.plot(xs[-1], ys[-1], "d", color=C_RSIM, ms=6, zorder=5)

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3, lw=0.6)
        ax.set_xlabel("x campo (m)"); ax.set_ylabel("y campo (m)")
        ax.set_title(f"{p['event']}  (t0={p['t0_time']:.2f}s, desl={p['disp']*100:.0f} cm)",
                     fontsize=10)
        # anotacao: erro FINAL twin/grSim (cm) + horizonte
        txt = (f"erro final @ {horizon_s:.1f}s:\n"
               f"twin = {p['err_twin']:.1f} cm\n"
               f"grSim = {p['err_rsim']:.1f} cm")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Trajetória: robô real vs twin (FOPDT) vs grSim (ODE)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figdir = repo_root() / "results" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    base = figdir / f"meeting_overlay_{stamp}"
    png, svg = f"{base}.png", f"{base}.svg"
    fig.savefig(png, dpi=200); fig.savefig(svg)
    plt.close(fig)

    print("=" * 72)
    print("OVERLAY DE TRAJETORIA — real vs twin (FOPDT) vs grSim (ODE)")
    print(f"hold-out: {n_seg} segmentos (idle dropped={n_drop}); "
          f"H_max={h_max} ({horizon_s:.2f}s); modo A (warm-up Lw={LW}, sub_ms={SUB_MS})")
    print("=" * 72)
    print(f"{'painel':>6} {'segmento':28} {'t0(s)':>7} {'desl(cm)':>9} "
          f"{'twin(cm)':>9} {'grSim(cm)':>10}")
    for i, p in enumerate(panels):
        print(f"{i+1:>6} {str(p['event'])[:28]:28} {p['t0_time']:>7.2f} "
              f"{p['disp']*100:>9.1f} {p['err_twin']:>9.1f} {p['err_rsim']:>10.1f}")
    rr = repo_root()
    print(f"\nPNG: {png}")
    print(f"SVG: {svg}")
    return panels, png, svg


if __name__ == "__main__":
    render()

"""Figura de fit HOLD-OUT de vx/vy (overlay medido x simulado P1D)

Reproducao EM LEITURA do pipeline de automation/identify_trans_dof.m
(load_uy_trans), sem re-identificar nada:
  - grade t = t0:Ts:t1 (t0=max inicio, t1=min fim de commands/processed_robots),
    Ts=1/60 (matlab_colon de phase0_replay.resample — mesma semantica do MATLAB);
  - u = interp1(t_cmd, move_x|y, t, 'previous', 0);
  - posicao da visao mm->m interp linear; theta unwrap antes de interpolar;
  - Savitzky-Golay win=7 ordem=2 em px,py (scipy.signal.savgol_filter, como
    sgolayfilt); vel = gradient/Ts; rotacao mundo->corpo por theta;
  - simulacao do P1D promovido (config/twin_params.json) em grade fina
    (Ts/20): atraso Td por shift 'previous', lag 1a ordem ZOH-exato, ganho K;
  - condicao inicial do lag estimada por minimos quadrados (espelha o default
    'InitialCondition' estimado do compare() do MATLAB); fit tambem reportado
    com IC zero para transparencia;
  - fit NRMSE = 100*(1 - ||y-yhat|| / ||y-mean(y)||)  (formula do compare()).

Eventos de HOLD-OUT (identicos aos de sysid_vxvy_v2, seed 42 — confirmados
bit-a-bit em results/reidentificacao_20260724.md, PARTE A):
  vx: front_to_back_01_01 (fit MATLAB 86.7%)
  vy: side_to_side_01_05 (86.8), _01_06 (86.3), _01_07 (87.3), _02_03 (88.1)


Uso:  python3 scripts/fig_fit_holdout_vxvy.py
Saida: results/figs/fit_holdout_vxvy_<stamp>.{png,pdf}
       results/fit_holdout_vxvy_<stamp>.csv
       latex/figures/fit_holdout_vxvy.png   (candidato; so se gate passar)
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phase0_replay.resample import matlab_colon, interp_previous  # noqa: E402

TS = 1.0 / 60.0
SG_WIN, SG_ORDER = 7, 2
TEAM = "allies"
DATASET = "2026-05-18_19-2-15"
UPSAMPLE = 20            # grade fina p/ atraso fracionario + lag ZOH-exato
TOL_PP = 5.0             # gate de divergencia (pontos percentuais)

# (dof, label, evento, fit MATLAB de referencia — results/reidentificacao_20260724.md)
EVENTS = [
    ("vx", "front_to_back", "front_to_back_01_01", 86.7),
    ("vy", "side_to_side", "side_to_side_01_05", 86.8),
    ("vy", "side_to_side", "side_to_side_01_06", 86.3),
    ("vy", "side_to_side", "side_to_side_01_07", 87.3),
    ("vy", "side_to_side", "side_to_side_02_03", 88.1),
]


def load_uy(label: str, event: str, dof: str):
    """Porte em leitura de load_uy_trans (identify_trans_dof.m)."""
    d = REPO / "data" / "extracted" / DATASET / "splits" / label / event
    C = pd.read_parquet(d / "commands.parquet")
    rid = int(C["robot_id"].mode().iloc[0])
    C = C[C["robot_id"].astype("int64") == rid].sort_values("timestamp_event", kind="stable")
    R = pd.read_parquet(d / "processed_robots.parquet")
    R = R[(R["team"] == TEAM) & (R["robot_id"].astype("int64") == rid)]
    R = R.sort_values("timestamp_event", kind="stable")
    tc = C["timestamp_event"].to_numpy(float)
    tr = R["timestamp_event"].to_numpy(float)
    t0, t1 = max(tc[0], tr[0]), min(tc[-1], tr[-1])
    t = matlab_colon(t0, t1, TS)
    cmdcol = "move_x" if dof == "vx" else "move_y"
    u = interp_previous(tc, C[cmdcol].to_numpy(float), t, 0.0)
    px = np.interp(t, tr, R["position_x"].to_numpy(float) / 1000.0)
    py = np.interp(t, tr, R["position_y"].to_numpy(float) / 1000.0)
    th = np.interp(t, tr, np.unwrap(R["position_w"].to_numpy(float)))
    win = min(SG_WIN, 2 * ((len(t) - 1) // 2) + 1)
    if win > SG_ORDER:
        px = savgol_filter(px, win, SG_ORDER)
        py = savgol_filter(py, win, SG_ORDER)
    vxw = np.gradient(px) / TS
    vyw = np.gradient(py) / TS
    if dof == "vx":
        y = np.cos(th) * vxw + np.sin(th) * vyw
    else:
        y = -np.sin(th) * vxw + np.cos(th) * vyw
    m = ~(np.isnan(u) | np.isnan(y))
    return t[m] - t[m][0], u[m], y[m]


def sim_p1d(t, u, K, tau, Td, x0=0.0):
    """P1D  K e^{-Td s}/(tau s + 1)  com entrada ZOH; grade fina Ts/UPSAMPLE.
    Atraso 'previous' (0 antes do inicio), lag ZOH-exato, estado inicial x0
    (unidades pre-ganho: y(0) = K*x0)."""
    n = len(t)
    dtf = TS / UPSAMPLE
    nf = (n - 1) * UPSAMPLE + 1
    uf = u[np.minimum(np.arange(nf) // UPSAMPLE, n - 1)]
    # entrada atrasada na grade fina (0 antes do inicio do experimento)
    shift = Td / dtf
    idx = np.floor(np.arange(nf) - shift + 1e-9).astype(int)
    ud = np.where(idx >= 0, uf[np.maximum(idx, 0)], 0.0)
    e = np.exp(-dtf / tau)
    v = np.empty(nf)
    v[0] = x0
    for j in range(1, nf):
        v[j] = e * v[j - 1] + (1.0 - e) * ud[j - 1]
    return K * v[::UPSAMPLE]


def fit_nrmse(y, yhat):
    return 100.0 * (1.0 - np.linalg.norm(y - yhat) / np.linalg.norm(y - np.mean(y)))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(REPO / "config" / "twin_params.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rows, panels = [], []
    for dof, label, event, fit_ref in EVENTS:
        p = cfg["dof"][dof]
        K, tau, Td = float(p["K"]), float(p["tau"]), float(p["Td"])
        t, u, y = load_uy(label, event, dof)
        y0hat = sim_p1d(t, u, K, tau, Td, x0=0.0)
        # IC do lag por minimos quadrados (espelha compare(): IC estimada)
        b = K * np.exp(-t / tau)
        r = y - y0hat
        x0 = float(np.dot(r, b) / np.dot(b, b))
        yhat = y0hat + x0 * b
        f_est = fit_nrmse(y, yhat)
        f_zero = fit_nrmse(y, y0hat)
        rows.append([dof, event, K, tau, Td, len(t), x0, f_est, f_zero, fit_ref,
                     f_est - fit_ref])
        panels.append((dof, event, t, y, yhat, f_est, fit_ref))
        print(f"{dof} {event}: fit(IC est.)={f_est:.1f}%  fit(IC 0)={f_zero:.1f}%  "
              f"MATLAB={fit_ref:.1f}%  delta={f_est - fit_ref:+.1f} pp")

    ok = all(abs(r[-1]) <= TOL_PP for r in rows)

    # ---- CSV ----
    csvp = REPO / "results" / f"fit_holdout_vxvy_{stamp}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dof", "evento", "K", "tau_s", "Td_s", "n_amostras",
                    "x0_lag_estimado", "fit_py_ic_estimada_pct", "fit_py_ic_zero_pct",
                    "fit_matlab_ref_pct", "delta_pp"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}",
                        r[5], f"{r[6]:.6f}", f"{r[7]:.2f}", f"{r[8]:.2f}",
                        f"{r[9]:.1f}", f"{r[10]:+.2f}"])

    # ---- figura IEEE column-width, legivel em P&B ----
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7})
    fig, axes = plt.subplots(len(panels), 1, figsize=(3.5, 7.6), sharex=False)
    for ax, (dof, event, t, y, yhat, f_est, fit_ref) in zip(axes, panels):
        ax.plot(t, y, "-", color="0.55", lw=1.0, label="measured")
        ax.plot(t, yhat, "--", color="black", lw=1.2, label="simulated P1D")
        ax.set_ylabel(f"$v_{{{dof[1]}}}$ (m/s)")
        ax.set_title(f"{dof}: {event} — fit {f_est:.1f}%", loc="left", fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        ax.margins(x=0.01)
    axes[0].legend(loc="best", frameon=True, framealpha=0.9)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout(h_pad=0.7)
    figbase = REPO / "results" / "figs" / f"fit_holdout_vxvy_{stamp}"
    fig.savefig(f"{figbase}.png", dpi=300)
    fig.savefig(f"{figbase}.pdf")
    plt.close(fig)

    print(f"\nCSV:    {csvp}")
    print(f"Figura: {figbase}.png / .pdf")
    if ok:
        dst = REPO / "latex" / "figures" / "fit_holdout_vxvy.png"
        shutil.copyfile(f"{figbase}.png", dst)
        print(f"GATE OK (todas |delta| <= {TOL_PP} pp) -> candidato copiado: {dst}")
    else:
        print(f"GATE FALHOU (|delta| > {TOL_PP} pp em algum evento) -> "
              "figura NAO copiada para latex/figures/; reportar divergencia.")


if __name__ == "__main__":
    main()

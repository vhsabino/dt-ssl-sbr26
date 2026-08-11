"""Percurso completo do segmento SQUARE — FORA DO HOLD-OUT (visualizacao).

Diagnostico: 'square' NAO esta em config/holdout_events.json (situacao b). Ele vive
na MESMA fonte canonica (data/extracted/<dataset>/splits/square/square_*) e e
carregado pelo MESMO loader (io.load_segment). Square foi usado na identificacao
(run_bayesopt_*), logo NAO e hold-out no sentido estrito => rotulado
"fora do hold-out (visualizacao)".

Reusa EXATAMENTE o codigo de percurso_completo (free-run de segmento inteiro,
twin azul / grSim vermelho / real preto, marcador 1.5 s, destaque de rotacao,
painel erro-vs-tempo). Params do twin inalterados (vx/vy promovidos, w placeholder)
=> espera-se o twin entortar nos 4 cantos (~90 graus) por causa do w placeholder;
o destaque de rotacao acende nos cantos. Isso e a historia por-DOF, nao bug.

Uso:
  ~/miniforge3/envs/rsim310/bin/python -m phase0_replay.percurso_square
"""
from __future__ import annotations

import numpy as np

from .io import repo_root, load_segment, event_dir
from .plant import AnalyticFOPDTPlant
from .rsim_plant import RSimReplay
from .params import load_twin_params
from .eval_twin_vs_rsim import _silence_engine_stdout
from .percurso_completo import (_full_freerun, _per_segment_fig, _grid_fig,
                                _tortuosity, SUB_MS, LW, W_HI)
from datetime import datetime

LABEL = "square"
SQUARE_EVENTS = [f"square_0{b}_0{i}" for b in (1, 2) for i in range(1, 7)]
OUT_LABEL = "fora do hold-out (visualizacao)"


def _bbox_area(R):
    x, y = R[:, 0], R[:, 1]
    return float((x.max() - x.min()) * (y.max() - y.min()))


def run():
    import matplotlib
    matplotlib.use("Agg")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figdir = repo_root() / "results" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    params = load_twin_params()

    # carrega todos os square que existirem (mesmo loader canonico)
    segs = []
    for ev in SQUARE_EVENTS:
        d = event_dir(LABEL, ev)
        if not (d / "commands.parquet").exists():
            continue
        try:
            segs.append(load_segment(LABEL, ev))
        except Exception as e:  # noqa: BLE001 — reporta, nao engole
            print(f"  [aviso] falhou carregar {ev}: {type(e).__name__}: {e}")
    if not segs:
        raise SystemExit("nenhum segmento square carregavel encontrado")

    # free-runs (TODOS os squares): twin sem silenciar; grSim silenciando o engine.
    twin_runs = [_full_freerun(s, AnalyticFOPDTPlant(params), params) for s in segs]
    with _silence_engine_stdout():
        rsim_runs = [_full_freerun(s, RSimReplay(warmup=True, sub_ms=SUB_MS), params)
                     for s in segs]

    items, per_files = [], []
    for s, tw, rs in zip(segs, twin_runs, rsim_runs):
        name = str(s.meta["event"])
        T, R, t_rel, w_path, _ = tw
        S = rs[0]                                  # pose_sim do grSim (mesmas R/t_rel)
        png, dur, et_f, es_f = _per_segment_fig(name, R, T, S, t_rel, w_path, figdir,
                                                stamp, extra_label=OUT_LABEL)
        per_files.append(png)
        tort, plen, net = _tortuosity(R)
        items.append(dict(name=name, R=R, T=T, S=S, w=w_path, dur=dur, et_f=et_f,
                          es_f=es_f, tort=tort, plen=plen, net=net,
                          area=_bbox_area(R), whi=float(np.mean(w_path >= W_HI))))

    # grid com TODOS os squares (mesmo render, titulo rotulado fora do hold-out)
    grid_png, grid_svg = _grid_fig(
        items, figdir, stamp,
        suptitle=("Percurso completo (free-run open-loop) — TODOS os SQUARE "
                  f"[{OUT_LABEL}]\nVISUALIZACAO, nao metrica; alem do 'x' (1.5 s) "
                  "e extrapolacao"),
        fname="percurso_completo_grid_square")

    print("=" * 92)
    print(f"SQUARE — TODOS os {len(items)} segmentos ({OUT_LABEL})")
    print("erro final NAO e metrica; o numero validado e o de 1.5 s (marcador x).")
    print(f"modo A (Lw={LW}); limiar rotacao |w_cmd|>={W_HI} rad/s")
    print("=" * 92)
    print(f"{'evento':16} {'dur(s)':>7} {'twin_fin(cm)':>12} {'grSim_fin(cm)':>13} "
          f"{'grSim/twin':>11} {'bbox(m2)':>9} {'%|w|>=' + str(W_HI):>8}")
    for it in sorted(items, key=lambda x: x["area"], reverse=True):
        ratio = it["es_f"] / it["et_f"] if it["et_f"] > 1e-6 else float("inf")
        print(f"{it['name']:16} {it['dur']:>7.1f} {it['et_f']:>12.1f} {it['es_f']:>13.1f} "
              f"{ratio:>10.1f}x {it['area']:>9.3f} {it['whi']*100:>7.0f}%")
    wmax = max(it["whi"] for it in items)
    print(f"\nNota honesta: |w_cmd|>={W_HI} rad/s atinge no max {wmax*100:.0f}% das amostras "
          "=> squares sao HOLONOMICOS (strafe), heading quase fixo. O drift do twin\n"
          "aqui e TRANSLACIONAL cumulativo (open-loop), NAO artefato do w placeholder; "
          "o destaque de rotacao fica apagado de proposito (fiel ao dado).")
    print("\n--- ARQUIVOS ---")
    for p in per_files:
        print(f"  {p}")
    print(f"  {grid_png}")
    print(f"  {grid_svg}")
    return per_files, grid_png


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
eval_rsim_delay_ablation.py — ablação "rSim + atraso de transporte casado".

MOTIVO (revisor B5/B6): a comparação gêmeo-vs-rSim tem UMA assimetria
estrutural — só o gêmeo carrega um atraso de transporte explícito. Parte do gap
de curto horizonte pode vir do termo de atraso, não da dinâmica identificada.
Esta ablação limita essa parcela: aplica o Td do gêmeo à SAÍDA do rSim como
deslocamento puro (pose_out[h] = pose_rsim[max(0, h-m)]), SEM re-simular.

DESENHO
-------
Uma única passada do rSim produz as trajetórias; a versão atrasada é derivada
por deslocamento das mesmas trajetórias. Isso (a) torna a ablação exata em vez
de aproximada, e (b) PERSISTE as trajetórias do rSim, que nunca foram salvas
(pendência 1 de docs/BACKLOG_EXTENSAO_JINT.md).

m = ceil(Td_trans / dt), com Td_trans = média de Td_vx e Td_vy. A pose é um
objeto conjunto, então não há como aplicar um atraso por DOF a ela; como a
alegação em disputa é translacional, usa-se o atraso translacional médio.

USO (o rollout do rSim leva ~45 s no total; o cache permite fatiar)
    python3 scripts/eval_rsim_delay_ablation.py --cache          # repetir até "COMPLETO"
    python3 scripts/eval_rsim_delay_ablation.py --finalize

Somente leitura sobre data/ e models/; grava artefatos NOVOS em results/.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase0_replay.eval_twin_vs_rsim import _silence_engine_stdout  # noqa: E402
from phase0_replay.splits import load_holdout_segments  # noqa: E402
from phase0_replay.params import load_twin_params  # noqa: E402
from phase0_replay.metrics import load_metrics_config, evaluate  # noqa: E402
from phase0_replay.plant import AnalyticFOPDTPlant  # noqa: E402
from phase0_replay.rollout import (DEFAULT_HMAX, DEFAULT_STRIDE,  # noqa: E402
                                   rollout_segment)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "_cache_rsim_rollouts.pkl"
LW = 15


def _cache_load() -> dict:
    if CACHE.is_file():
        with CACHE.open("rb") as f:
            return pickle.load(f)
    return {}


def _cache_save(d: dict) -> None:
    with CACHE.open("wb") as f:
        pickle.dump(d, f, protocol=4)


def do_cache(budget_s: float) -> None:
    """Roda rollouts do rSim segmento a segmento até estourar o orçamento."""
    from phase0_replay.rsim_plant import RSimReplay

    mcfg = load_metrics_config()
    horizons = list(mcfg["horizons"])
    segs, _ = load_holdout_segments(min_displacement_m=0.1)
    cache = _cache_load()
    t_start = time.time()
    done_now = 0

    for i, s in enumerate(segs):
        key = f"{i}"
        if key in cache:
            continue
        if time.time() - t_start > budget_s:
            break
        with _silence_engine_stdout():
            ro = rollout_segment(s, RSimReplay(warmup=True, sub_ms=1),
                                 h_max=DEFAULT_HMAX, stride=DEFAULT_STRIDE,
                                 lookback=LW, horizons=horizons)
        cache[key] = ro
        done_now += 1
        _cache_save(cache)

    total = len(segs)
    print(f"cache: {len(cache)}/{total} segmentos (+{done_now} nesta chamada)")
    if len(cache) >= total:
        print("COMPLETO — rode com --finalize")


def _shift(ro, m: int):
    """Aplica atraso puro de m amostras à SAÍDA (pose_sim) de cada janela."""
    out = copy.deepcopy(ro)
    for w in out.windows:
        ps = w.pose_sim
        shifted = np.empty_like(ps)
        for h in range(ps.shape[0]):
            shifted[h] = ps[max(0, h - m)]
        w.pose_sim = shifted
    return out


def do_finalize() -> None:
    params = load_twin_params()
    mcfg = load_metrics_config()
    horizons = list(mcfg["horizons"])
    segs, _ = load_holdout_segments(min_displacement_m=0.1)
    cache = _cache_load()
    if len(cache) < len(segs):
        sys.exit(f"cache incompleto ({len(cache)}/{len(segs)}); rode --cache")

    rsim_ros = [cache[str(i)] for i in range(len(segs))]

    dt = params["ts"] if isinstance(params, dict) and "ts" in params else 1.0 / 60.0
    td_tr = 0.5 * (params["dof"]["vx"]["Td"] + params["dof"]["vy"]["Td"])
    m = math.ceil(td_tr / dt)
    print(f"Td_trans medio = {td_tr:.5f} s ; dt = {dt:.5f} s ; m = {m} amostras "
          f"({m*dt*1000:.1f} ms aplicados)")

    twin_ros = [rollout_segment(s, AnalyticFOPDTPlant(params), h_max=DEFAULT_HMAX,
                                stride=DEFAULT_STRIDE, lookback=LW,
                                horizons=horizons) for s in segs]
    abl_ros = [_shift(r, m) for r in rsim_ros]

    res_raw = evaluate(twin_ros, rsim_ros, params=params, mcfg=mcfg)
    res_abl = evaluate(twin_ros, abl_ros, params=params, mcfg=mcfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    lines = [f"# Ablação rSim + atraso casado — {stamp}", "",
             f"Atraso aplicado: m = {m} amostras ({m*dt*1000:.1f} ms), "
             f"Td_trans medio = {td_tr*1000:.1f} ms.", "",
             "| h (s) | twin | rSim | gap rSim | rSim+Td | gap rSim+Td | delta |",
             "|---|---|---|---|---|---|---|"]
    a, b = res_raw["scopes"]["trans"], res_abl["scopes"]["trans"]
    ts = res_raw["table"].ts
    for k, h in enumerate(a["horizons"]):
        lines.append(
            f"| {h*ts:.2f} | {a['epos_twin_med'][k]:.4f} | {a['epos_def_med'][k]:.4f} "
            f"| {a['gap_epos'][k]:+.4f} | {b['epos_def_med'][k]:.4f} "
            f"| {b['gap_epos'][k]:+.4f} | "
            f"{a['gap_epos'][k]-b['gap_epos'][k]:+.4f} |")
    lines += ["", f"N_indep = {a['n_indep']}", "",
              "`delta` = quanto do gap do rSim é explicado pelo termo de atraso."]
    out = ROOT / "results" / f"eval_rsim_delay_ablation_{stamp}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nartefato: results/{out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--budget", type=float, default=32.0)
    a = ap.parse_args()
    if a.cache:
        do_cache(a.budget)
    elif a.finalize:
      
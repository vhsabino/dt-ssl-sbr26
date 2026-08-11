"""Compara o TWIN (AnalyticFOPDTPlant) vs o RSimReplay (rSoccer/rSim, ODE) nas
MESMAS janelas, MESMOS gates e MESMA inferencia (Wilcoxon nao-sobreposto +
moving-block bootstrap) do pipeline do twin-vs-ideal. Reusa
phase0_replay.metrics.evaluate INTEGRALMENTE (so a apresentacao e local, para
rotular 'rSim' no lugar de 'ideal').

O bin de TRANSLACAO e o veredito real desta rodada (vx/vy identificados; w e
placeholder => misto/rot ficam INDETERMINADOS pelo gate de omega).

Justica de IC (o rSim nao seta velocidade):
  modo A (default) = warm-up por janela (Lw amostras) + re-ancora rigida SE(2).
  modo B = sem warm-up (rSim parte do repouso) e DESCARTA as K primeiras amostras
           da metrica para AMBOS os plants (verificacao de robustez).

Uso:
  python -m phase0_replay.eval_twin_vs_rsim                 # modo A, Lw=15
  python -m phase0_replay.eval_twin_vs_rsim --mode B --drop-k 15
  python -m phase0_replay.eval_twin_vs_rsim --lw 12 --sub-ms 1
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .io import repo_root
from .splits import load_holdout_segments
from .plant import AnalyticFOPDTPlant
from .rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
from .metrics import evaluate, format_bin_distribution, load_metrics_config, BIN_LABEL, BIN_ORDER
from .params import load_twin_params


@contextlib.contextmanager
def _silence_engine_stdout():
    """Redireciona o fd 1 (printf do engine C++ do rc-robosim) p/ /dev/null."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _build_rollouts(segs, plant_factory, *, h_max, stride, lookback, horizons,
                    silence=False):
    ros = []
    ctx = _silence_engine_stdout() if silence else contextlib.nullcontext()
    with ctx:
        for s in segs:
            plant = plant_factory()
            ros.append(rollout_segment(s, plant, h_max=h_max, stride=stride,
                                       lookback=lookback, horizons=horizons))
    return ros


# --------------------------------------------------------------------------
def _figures(result, outdir, stamp, def_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = []
    C_TWIN, C_DEF = (0.10, 0.30, 0.65), (0.75, 0.25, 0.15)
    for scope in ("all",) + BIN_ORDER:
        res = result["scopes"][scope]
        if res.get("n_total", 0) == 0 or "epos_twin_med" not in res:
            continue
        hs = res["horizons_s"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, key, ttl, ylab in ((ax[0], "epos", "RMSE posicao", "e_pos mediana (m)"),
                                   (ax[1], "eang", "erro heading", "e_ang mediana (rad)")):
            tw, df = res[f"{key}_twin_med"], res[f"{key}_def_med"]
            tw_ci, df_ci = res[f"{key}_twin_ci"], res[f"{key}_def_ci"]
            a.fill_between(hs, df_ci[:, 0], df_ci[:, 1], color=C_DEF, alpha=0.18)
            a.fill_between(hs, tw_ci[:, 0], tw_ci[:, 1], color=C_TWIN, alpha=0.18)
            a.plot(hs, df, "-o", color=C_DEF, lw=1.8, label=def_label)
            a.plot(hs, tw, "-d", color=C_TWIN, lw=2.0, label="twin (FOPDT)")
            a.set_xlabel("horizonte h (s)"); a.set_ylabel(ylab); a.set_title(ttl)
            a.grid(alpha=0.3); a.legend(loc="upper left")
        vb = result["verdicts"].get(scope)
        vtxt = f" — {vb['verdict']}" if vb else ""
        fig.suptitle(f"twin vs {def_label} [{BIN_LABEL.get(scope,'TODOS')}] "
                     f"(N={res['n_total']}, indep={res['n_indep']}){vtxt}")
        fig.tight_layout()
        base = outdir / f"rsim_twin_{scope}_{stamp}"
        fig.savefig(f"{base}.png", dpi=150); fig.savefig(f"{base}.svg")
        plt.close(fig)
        files.append(f"{base}.png")
    return files


def _markdown(result, meta, fig_files):
    mcfg = result["mcfg"]; href = mcfg["verdict"]["h_ref"]; ts = result["table"].ts
    L = [f"# twin (FOPDT) vs {meta['def_label']} — metrica SE(2) multi-horizonte",
         "",
         f"Modo de justica de IC: **{meta['mode_desc']}**. "
         f"Lw(warm-up)={meta['lw']} amostras, sub_ms={meta['sub_ms']}, "
         f"stride={meta['stride']}, h_max={meta['h_max']}.",
         f"Config metrica: horizontes(amostras)={list(result['table'].horizons)} "
         f"(s={[round(float(x),3) for x in result['table'].horizons*ts]}), "
         f"h_ref={href}, N_min={mcfg['verdict']['n_min_independent']}, "
         f"alpha={mcfg['verdict']['alpha']}, effect_min={mcfg['verdict']['effect_size_min']}, "
         f"bloco_MBB={result['table'].block_windows} janelas.",
         "", "## Distribuicao de bins", "",
         "| bin | total (densas) | independentes |", "|---|---:|---:|"]
    d = result["bin_distribution"]
    for b in BIN_ORDER:
        L.append(f"| {BIN_LABEL[b]} | {d[b]['total']} | {d[b]['indep']} |")
    L.append(f"| **TODOS** | {d['all']['total']} | {d['all']['indep']} |")

    for scope in ("all",) + BIN_ORDER:
        res = result["scopes"][scope]
        L += ["", f"## {BIN_LABEL.get(scope,'TODOS (agregado)')}"]
        if res.get("n_total", 0) == 0 or "epos_twin_med" not in res:
            L.append("_(sem janelas)_"); continue
        L.append(f"N total={res['n_total']}, independentes={res['n_indep']}")
        L += ["", f"| h (s) | twin e_pos | {meta['def_label']} e_pos | gap (m) | gap CI95 | "
              "p (nao-sobr.) | effect |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for k, h in enumerate(res["horizons"]):
            cik = res["gap_epos_ci"][k]
            L.append(f"| {h*ts:.2f} | {res['epos_twin_med'][k]:.4f} | {res['epos_def_med'][k]:.4f} "
                     f"| {res['gap_epos'][k]:+.4f} | [{cik[0]:+.4f},{cik[1]:+.4f}] "
                     f"| {res['p_wilcoxon'][k]:.3g} | {res['effect'][k]:+.3f} |")
        if res.get("divergence"):
            L.append("\n> DIVERGENCIA: Wilcoxon significativo mas bootstrap CI inclui 0 em algum h.")

    L += ["", "## Vereditos por bin (gates: omega placeholder + populacao N_min)", "",
          "| bin | veredito | N indep | gap@h_ref | p@h_ref | effect@h_ref |",
          "|---|---|---:|---:|---:|---:|"]
    for b in BIN_ORDER:
        v = result["verdicts"][b]
        g = f"{v.get('gap_at_href'):+.4f}" if v.get("gap_at_href") is not None else "-"
        pp = f"{v.get('p_at_href'):.3g}" if v.get("p_at_href") is not None else "-"
        ee = f"{v.get('effect_at_href'):+.3f}" if v.get("effect_at_href") is not None else "-"
        L.append(f"| {BIN_LABEL[b]} | {v['verdict']} | {v['n_indep']} | {g} | {pp} | {ee} |")
    if fig_files:
        L += ["", "## Figuras"] + [f"- `{f}`" for f in fig_files]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
def run(*, mode="A", lw=15, sub_ms=1, drop_k=15, stride=DEFAULT_STRIDE,
        h_max=DEFAULT_HMAX):
    from .rsim_plant import RSimReplay
    params = load_twin_params()
    mcfg = load_metrics_config()
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)
    horizons_full = list(mcfg["horizons"])

    warmup = (mode == "A")
    mcfg_eval = dict(mcfg)
    if mode == "B":
        mcfg_eval = dict(mcfg)
        mcfg_eval["horizons"] = [h for h in horizons_full if h >= drop_k]
        mode_desc = f"B (sem warm-up; descarta amostras < {drop_k} p/ AMBOS)"
    else:
        mode_desc = f"A (warm-up Lw={lw} + re-ancora rigida SE(2))"

    twin_ros = _build_rollouts(segs, lambda: AnalyticFOPDTPlant(params),
                               h_max=h_max, stride=stride, lookback=lw,
                               horizons=horizons_full, silence=False)
    rsim_ros = _build_rollouts(segs, lambda: RSimReplay(warmup=warmup, sub_ms=sub_ms),
                               h_max=h_max, stride=stride, lookback=lw,
                               horizons=horizons_full, silence=True)

    result = evaluate(twin_ros, rsim_ros, params=params, mcfg=mcfg_eval)

    def_label = "rSim (ODE replay)"
    print("=" * 72)
    print(f"twin (FOPDT) vs {def_label} | modo {mode_desc} | Lw={lw} sub_ms={sub_ms}")
    print(f"segmentos hold-out={len(segs)} (dropped idle={len(dropped)})")
    print("=" * 72)
    print(format_bin_distribution(result["table"]))
    print("=" * 72)
    print("\nVereditos por bin:")
    for b in BIN_ORDER:
        v = result["verdicts"][b]
        extra = ""
        if v.get("gap_at_href") is not None:
            extra = (f" | gap@h_ref={v['gap_at_href']:+.4f} m, p={v['p_at_href']:.3g}, "
                     f"effect={v['effect_at_href']:+.3f}")
        print(f"  {BIN_LABEL[b]:14s}: {v['verdict']:34s} (N_indep={v['n_indep']}){extra}")

    # foco no bin de translacao (o veredito real)
    rt = result["scopes"]["trans"]
    if rt.get("n_total"):
        print("\n--- BIN DE TRANSLACAO (veredito real) ---")
        H = rt["horizons"]; ts = result["table"].ts
        print(f"  N total={rt['n_total']}, N independentes={rt['n_indep']}")
        for k, h in enumerate(H):
            cik = rt["gap_epos_ci"][k]
            boot = "exclui0" if cik[0] > 0 else "inclui0"
            print(f"   h={h*ts:.2f}s: twin={rt['epos_twin_med'][k]:.4f} "
                  f"rsim={rt['epos_def_med'][k]:.4f} gap={rt['gap_epos'][k]:+.4f}m "
                  f"CI[{cik[0]:+.4f},{cik[1]:+.4f}]({boot}) p={rt['p_wilcoxon'][k]:.3g} "
                  f"eff={rt['effect'][k]:+.3f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    figdir = repo_root() / "results" / "figs"
    figs = _figures(result, figdir, stamp, def_label)
    meta = dict(def_label=def_label, mode_desc=mode_desc, lw=lw, sub_ms=sub_ms,
                stride=stride, h_max=h_max)
    md = _markdown(result, meta, [Path(f).name for f in figs])
    rep = repo_root() / "results" / f"eval_twin_vs_rsim_mode{mode}_{stamp}.md"
    rep.write_text(md, encoding="utf-8")
    print(f"\nFiguras: {len(figs)} em results/figs/ (rsim_twin_*_{stamp}.png/.svg)")
    print(f"Relatorio: {rep.relative_to(repo_root())}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--lw", type=int, default=15, help="warm-up window (amostras)")
    ap.add_argument("--sub-ms", type=int, default=1, dest="sub_ms")
    ap.add_argument("--drop-k", type=int, default=15, dest="drop_k",
                    help="modo B: descarta horizontes < K amostras")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--h-max", type=int, default=DEFAULT_HMAX, dest="h_max")
    a = ap.parse_args()
    run(mode=a.mode, lw=a.lw, sub_ms=a.sub_ms, drop_k=a.drop_k, stride=a.stride,
        h_max=a.h_max)


if __name__ == "__main__":
    main()

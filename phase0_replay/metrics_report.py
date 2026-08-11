"""Roda a metrica no hold-out: twin (AnalyticFOPDTPlant) vs default (IdealKinematic),
mesmas janelas (pareadas). Imprime a DISTRIBUICAO DE BINS primeiro, depois o
veredito por bin; salva figuras (PNG/SVG) e um relatorio Markdown.

Uso:  python -m phase0_replay.metrics_report
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .io import repo_root
from .splits import load_holdout_segments
from .plant import AnalyticFOPDTPlant, IdealKinematicPlant
from .rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
from .metrics import (evaluate, format_bin_distribution, make_figures,
                      markdown_report, load_metrics_config, BIN_LABEL)
from .params import load_twin_params


def run(stride: int = DEFAULT_STRIDE, h_max: int = DEFAULT_HMAX) -> dict:
    params = load_twin_params()
    mcfg = load_metrics_config()
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)

    twin = AnalyticFOPDTPlant(params)
    ideal = IdealKinematicPlant()
    twin_ros, def_ros = [], []
    for s in segs:
        twin_ros.append(rollout_segment(s, twin, h_max=h_max, stride=stride,
                                        horizons=mcfg["horizons"]))
        def_ros.append(rollout_segment(s, ideal, h_max=h_max, stride=stride,
                                       horizons=mcfg["horizons"]))

    result = evaluate(twin_ros, def_ros, params=params, mcfg=mcfg)

    # 1) DISTRIBUICAO DE BINS — antes de qualquer veredito
    print("=" * 70)
    print(format_bin_distribution(result["table"]))
    print("=" * 70)

    # 2) vereditos por bin
    print("\nVereditos por bin (gates: placeholder de omega + populacao N_min):")
    for b in ("trans", "misto", "rot"):
        v = result["verdicts"][b]
        extra = ""
        if v.get("gap_at_href") is not None:
            extra = (f" | gap@h_ref={v['gap_at_href']:+.4f} m, p={v['p_at_href']:.3g}, "
                     f"effect={v['effect_at_href']:+.3f}")
        print(f"  {BIN_LABEL[b]:14s}: {v['verdict']:34s} (N_indep={v['n_indep']}){extra}")

    # 3) figuras + relatorio
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    figdir = repo_root() / "results" / "figs"
    figs = make_figures(result, figdir, stamp)
    md = markdown_report(result, fig_files=[Path(f).name for f in figs])
    rep = repo_root() / "results" / f"phase0_metrics_report_{stamp}.md"
    rep.write_text(md, encoding="utf-8")
    print(f"\nFiguras: {len(figs)} salvas em results/figs/ (M1_metrics_*_{stamp}.png/.svg)")
    print(f"Relatorio: {rep.relative_to(repo_root())}")
    return result


if __name__ == "__main__":
    run()

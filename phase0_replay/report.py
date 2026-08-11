"""Relatorio do loader: sanidade dos streams crus + distribuicao de DURACAO dos
segmentos de hold-out (para fixar H_max nas metricas multi-horizonte).

Uso:  python -m phase0_replay.report
"""
from __future__ import annotations

import numpy as np

from .io import (segment_displacement_m, segment_peak_cmd_speed, event_dir)
from .splits import load_holdout_list, load_holdout_segments
from .validate import check_velocity_units, raw_stream_report


def duration_distribution(segments) -> dict:
    durs = np.array([s.duration_s for s in segments], dtype=float)
    ns = np.array([s.n for s in segments], dtype=int)
    return {
        "n_segments": int(durs.size),
        "min_s": float(durs.min()), "p25_s": float(np.percentile(durs, 25)),
        "median_s": float(np.median(durs)), "mean_s": float(durs.mean()),
        "p75_s": float(np.percentile(durs, 75)), "max_s": float(durs.max()),
        "total_s": float(durs.sum()),
        "samples_min": int(ns.min()), "samples_max": int(ns.max()),
        "H_max_suggested_s": float(durs.min()),   # horizonte maximo cabivel em TODOS
    }


def main() -> None:
    cfg = load_holdout_list()
    print("=" * 74)
    print("phase0_replay - RELATORIO DO LOADER (hold-out %s)" % cfg.get("dataset"))
    print("=" * 74)

    # --- sanity de unidade de velocity_* (1 evento representativo) ---
    e0 = cfg["events"][0]
    d0 = event_dir(e0["label"], e0["event"], cfg["dataset"])
    vu = check_velocity_units(d0 / "processed_robots.parquet", team=cfg.get("team", "allies"))
    print("\n[velocity unit sanity em %s]" % e0["event"])
    print("  vel mediana da POSICAO   = %.3f m/s" % vu["median_speed_from_position_m_s"])
    print("  velocity_norm logado     = %.1f   (=> %.3f m/s se mm/s)"
          % (vu["median_velocity_norm_logged"], vu["median_velocity_logged_as_m_s_if_mm_s"]))
    print("  ratio logado/posicao     = %.3f  -> velocity_* em mm/s: %s"
          % (vu["ratio_logged_over_position"], vu["unit_is_mm_per_s"]))
    rc = raw_stream_report(d0 / "commands.parquet")
    rr = raw_stream_report(d0 / "processed_robots.parquet")
    print("  commands:         %.1f Hz (dt med %.4fs, p10 %.4f p90 %.4f), gaps>5x=%d, dup_ts=%d"
          % (rc["rate_hz"], rc["dt_median_s"], rc["dt_p10_s"], rc["dt_p90_s"],
             rc["n_gaps_gt_5x"], rc["n_duplicate_ts"]))
    print("  processed_robots: %.1f Hz (dt med %.4fs, p10 %.4f p90 %.4f), gaps>5x=%d, dup_ts=%d"
          % (rr["rate_hz"], rr["dt_median_s"], rr["dt_p10_s"], rr["dt_p90_s"],
             rr["n_gaps_gt_5x"], rr["n_duplicate_ts"]))

    # --- carrega segmentos do hold-out (so com movimento) ---
    segs, dropped = load_holdout_segments(min_displacement_m=0.1)
    print("\n[segmentos] %d carregados, %d descartados (idle)" % (len(segs), len(dropped)))
    for d in dropped:
        print("  - DROP %s (%s)" % (d["event"], d["reason"]))

    print("\n%-22s %6s %8s %9s %9s" % ("event", "N", "dur[s]", "disp[m]", "pk_cmd[m/s]"))
    print("-" * 60)
    for s in segs:
        print("%-22s %6d %8.2f %9.3f %9.3f"
              % (s.meta["event"], s.n, s.duration_s,
                 segment_displacement_m(s), segment_peak_cmd_speed(s)))

    dist = duration_distribution(segs)
    print("\n[DISTRIBUICAO DE DURACAO dos segmentos]")
    for k in ["n_segments", "min_s", "p25_s", "median_s", "mean_s", "p75_s",
              "max_s", "total_s", "samples_min", "samples_max", "H_max_suggested_s"]:
        print("  %-20s = %s" % (k, ("%.3f" % dist[k]) if isinstance(dist[k], float) else dist[k]))
    print("\n  => H_max (horizonte que cabe em TODOS os segmentos) ~ %.2f s" % dist["H_max_suggested_s"])


if __name__ == "__main__":
    main()

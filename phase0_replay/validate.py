"""Validacoes de dados crus (antes/alem do Segment canonico).

- check_velocity_units: confirma que velocity_* esta em mm/s (e nao m/s),
  comparando a magnitude logada com a velocidade derivada da POSICAO.
- raw_stream_report: estatisticas de dt e contagem de gaps de timestamp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MM_TO_M = 1.0 / 1000.0


def _robot_frame(robots_path, team="allies", robot_id=None) -> pd.DataFrame:
    R = pd.read_parquet(robots_path)
    R = R[R["team"] == team]
    if robot_id is not None:
        R = R[R["robot_id"].astype("int64") == int(robot_id)]
    elif R["robot_id"].nunique() == 1:
        pass
    else:
        rid = R["robot_id"].mode().iloc[0]
        R = R[R["robot_id"] == rid]
    return R.sort_values("timestamp_event").reset_index(drop=True)


def check_velocity_units(robots_path, team="allies", robot_id=None,
                         ratio_lo=0.2, ratio_hi=5.0) -> dict:
    """velocity_* esta em mm/s? Compara median(velocity_norm)*1e-3 [m/s] com a
    velocidade mediana derivada da posicao (mm->m). Se as duas baterem (ratio ~1),
    velocity_* e mm/s. Se velocity_* fosse m/s, o ratio seria ~1000."""
    R = _robot_frame(robots_path, team, robot_id)
    t = R["timestamp_event"].to_numpy(float)
    px = R["position_x"].to_numpy(float) * MM_TO_M
    py = R["position_y"].to_numpy(float) * MM_TO_M
    sp_pos = np.hypot(np.gradient(px, t), np.gradient(py, t))      # m/s (da posicao)
    med_pos = float(np.nanmedian(sp_pos))
    med_log_mm_s = float(np.nanmedian(R["velocity_norm"].to_numpy(float)))  # se mm/s
    med_log_as_ms = med_log_mm_s * MM_TO_M                          # interpretando mm/s
    ratio = med_log_as_ms / med_pos if med_pos > 1e-9 else np.inf
    is_mm_s = ratio_lo <= ratio <= ratio_hi
    return {
        "median_speed_from_position_m_s": med_pos,
        "median_velocity_norm_logged": med_log_mm_s,
        "median_velocity_logged_as_m_s_if_mm_s": med_log_as_ms,
        "ratio_logged_over_position": ratio,
        "unit_is_mm_per_s": bool(is_mm_s),
    }


def raw_stream_report(path, gap_factor=5.0) -> dict:
    """dt mediano/p10/p90 e numero de gaps (dt > gap_factor*mediana) num stream."""
    df = pd.read_parquet(path)
    t = np.sort(df["timestamp_event"].to_numpy(float))
    d = np.diff(t)
    med = float(np.median(d)) if d.size else float("nan")
    ngap = int(np.count_nonzero(d > gap_factor * med)) if d.size else 0
    ndup = int(np.count_nonzero(d == 0))
    return {
        "n": int(t.size),
        "dt_median_s": med, "rate_hz": (1.0 / med) if med > 0 else float("nan"),
        "dt_p10_s": float(np.percentile(d, 10)) if d.size else float("nan"),
        "dt_p90_s": float(np.percentile(d, 90)) if d.size else float("nan"),
        "span_s": float(t[-1] - t[0]) if t.size else 0.0,
        "n_gaps_gt_%gx" % gap_factor: ngap,
        "n_duplicate_ts": ndup,
    }

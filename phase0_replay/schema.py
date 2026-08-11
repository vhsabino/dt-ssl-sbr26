"""Schema canonico em memoria de um segmento (Segment) + validacao.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

TS_DEFAULT = 1.0 / 60.0


@dataclass
class Segment:
    # --- identidade / proveniencia ---
    meta: dict[str, Any]
    ts: float                      # passo da grade (s), tipicamente 1/60

    # --- tempo (grade uniforme, s, monotonico) ---
    t: np.ndarray                  # (N,)
    dt: np.ndarray                 # (N,) diff(t); dt[0] := ts por convencao

    # --- comandos (entrada), frame de CORPO, SI, na grade t (ZOH) ---
    cmd_vx: np.ndarray             # (N,) m/s
    cmd_vy: np.ndarray             # (N,) m/s
    cmd_w: np.ndarray              # (N,) rad/s

    # --- truth (referencia), frame de MUNDO, na grade t (linear; theta unwrap) ---
    pos_x: np.ndarray              # (N,) m
    pos_y: np.ndarray              # (N,) m
    theta: np.ndarray              # (N,) rad (unwrapped)

    # --- velocidade da visao reamostrada (frame de MUNDO, m/s) — p/ sanity ---
    vel_x_world: np.ndarray        # (N,) m/s
    vel_y_world: np.ndarray        # (N,) m/s

    # --- serie de velocidade de CORPO derivada da POSICAO (m/s, m/s, rad/s) ---
    # Estimador do loader (body_velocity_series); da o velb0 de cada janela do
    # rollout. NAO e a velocity_* logada. Calculada UMA vez na carga (desacoplada
    # da pose em runtime: a guarda anti-teacher-forcing depende disso).
    velb_x: np.ndarray             # (N,) m/s  (corpo)
    velb_y: np.ndarray             # (N,) m/s  (corpo)
    velb_w: np.ndarray             # (N,) rad/s

    # --- condicoes iniciais para Plant.reset ---
    pose0: tuple[float, float, float]          # (x0, y0, theta0)  m, m, rad
    velb0: tuple[float, float, float]          # (vx0, vy0, w0)    m/s, m/s, rad/s (CORPO) = serie em t0=0

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    def __repr__(self) -> str:
        ev = self.meta.get("event", "?")
        return (f"Segment(event={ev!r}, N={self.n}, dur={self.duration_s:.2f}s, "
                f"ts={self.ts:.5f}s)")


# Faixas fisicas plausiveis para validacao (robo SSL; campo Div B ~ 9x6 m).
RANGES = {
    "cmd_vx": (-5.0, 5.0),     # m/s
    "cmd_vy": (-5.0, 5.0),     # m/s
    "cmd_w": (-30.0, 30.0),    # rad/s
    "pos_x": (-7.0, 7.0),      # m
    "pos_y": (-7.0, 7.0),      # m
}


def validate_segment(seg: Segment, ts: float = TS_DEFAULT, rtol_dt: float = 1e-6,
                     raise_on_error: bool = True) -> list[str]:
    """Valida grade/NaN/unidades de um Segment. Retorna lista de problemas;
    se raise_on_error e houver problemas, levanta ValueError."""
    issues: list[str] = []
    arrays = {
        "t": seg.t, "cmd_vx": seg.cmd_vx, "cmd_vy": seg.cmd_vy, "cmd_w": seg.cmd_w,
        "pos_x": seg.pos_x, "pos_y": seg.pos_y, "theta": seg.theta,
        "vel_x_world": seg.vel_x_world, "vel_y_world": seg.vel_y_world,
        "velb_x": seg.velb_x, "velb_y": seg.velb_y, "velb_w": seg.velb_w,
    }

    # shapes consistentes
    n = seg.t.size
    for name, arr in arrays.items():
        if arr.shape != (n,):
            issues.append(f"shape de {name} = {arr.shape}, esperado ({n},)")
        if arr.dtype != np.float64:
            issues.append(f"dtype de {name} = {arr.dtype}, esperado float64")

    # NaN
    for name, arr in arrays.items():
        nnan = int(np.count_nonzero(np.isnan(arr)))
        if nnan:
            issues.append(f"{name} tem {nnan} NaN")

    # grade uniforme 1/60 (gaps)
    if n >= 2:
        d = np.diff(seg.t)
        if np.any(d <= 0):
            issues.append("t nao e estritamente crescente")
        max_dev = float(np.max(np.abs(d - ts))) if d.size else 0.0
        if max_dev > rtol_dt * ts + 1e-9:
            issues.append(f"grade nao uniforme: max|dt - {ts:.6f}| = {max_dev:.2e} s "
                          f"(possivel gap de timestamp)")

    # faixas fisicas
    for name, (lo, hi) in RANGES.items():
        arr = arrays[name]
        if arr.size and (np.nanmin(arr) < lo or np.nanmax(arr) > hi):
            issues.append(f"{name} fora de faixa [{lo},{hi}]: "
                          f"[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")

    if issues and raise_on_error:
        raise ValueError("Segment invalido (%s):\n  - %s"
                         % (seg.meta.get("event", "?"), "\n  - ".join(issues)))
    return issues

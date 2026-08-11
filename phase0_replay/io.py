"""Loader: dois parquets assincronos/irregulares -> Segment canonico em SI.

Reproduz a reamostragem do pipeline de ID (build_idData_per_dof):
  grade = matlab_colon(tc[0], tc[-1], 1/60); comandos ZOH; pose linear; theta unwrap.
Conversoes obrigatorias: position mm->m, velocity mm/s->m/s, theta unwrap.
Comando JA em corpo e SI: nao rotaciona, nao escala.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .resample import matlab_colon, interp_previous, interp_linear_nan
from .schema import Segment, TS_DEFAULT, validate_segment

DATASET_DEFAULT = "2026-05-18_19-2-15"
MM_TO_M = 1.0 / 1000.0


def repo_root(start: Path | None = None) -> Path:
    """Raiz do repo = primeiro ancestral com 'data' e 'config'."""
    cur = (start or Path(__file__)).resolve()
    for p in [cur, *cur.parents]:
        if (p / "data").is_dir() and (p / "config").is_dir():
            return p
    raise RuntimeError("raiz do repo nao encontrada (esperado conter data/ e config/)")


def event_dir(label: str, event: str, dataset: str = DATASET_DEFAULT,
              data_root: Path | None = None) -> Path:
    root = data_root or (repo_root() / "data" / "extracted")
    return Path(root) / dataset / "splits" / label / event


def _select_robot(C: pd.DataFrame, R: pd.DataFrame, team: str, robot_id: int | None):
    ids = pd.unique(C["robot_id"])
    if robot_id is None:
        if len(ids) != 1:
            raise ValueError(f"commands tem varios robot_id {list(map(int, ids))}; "
                             f"passe robot_id explicito")
        rid = int(ids[0])
    else:
        rid = int(robot_id)
    C = C[C["robot_id"].astype("int64") == rid].sort_values("timestamp_event", kind="stable")
    R = R[(R["team"] == team) & (R["robot_id"].astype("int64") == rid)].sort_values("timestamp_event", kind="stable")
    if len(C) < 2 or len(R) < 2:
        raise ValueError(f"sem amostras suficientes (robot_id={rid}, team={team}): "
                         f"commands={len(C)}, robots={len(R)}")
    return C, R, rid


def load_event(commands_path: str | Path, robots_path: str | Path, *,
               team: str = "allies", robot_id: int | None = None,
               ts: float = TS_DEFAULT, meta: dict[str, Any] | None = None,
               validate: bool = True) -> Segment:
    """Carrega um evento (caminhos dos dois parquets) -> Segment em SI na grade ts."""
    C = pd.read_parquet(commands_path)
    R = pd.read_parquet(robots_path)
    C, R, rid = _select_robot(C, R, team, robot_id)

    tc = C["timestamp_event"].to_numpy(dtype=float)
    tr = R["timestamp_event"].to_numpy(dtype=float)

    # grade uniforme 1/60 a partir do intervalo dos COMANDOS (igual ao ID)
    grid = matlab_colon(tc[0], tc[-1], ts)

    # comandos: JA em corpo e SI -> ZOH (previous), preenche 0 fora do dominio
    cmd_vx = interp_previous(tc, C["move_x"].to_numpy(float), grid, 0.0)
    cmd_vy = interp_previous(tc, C["move_y"].to_numpy(float), grid, 0.0)
    cmd_w  = interp_previous(tc, C["move_w"].to_numpy(float), grid, 0.0)

    # pose (mundo): mm->m, theta unwrap ANTES de interpolar; linear com NaN fora
    theta_uw = np.unwrap(R["position_w"].to_numpy(float))
    theta = interp_linear_nan(tr, theta_uw, grid)
    pos_x = interp_linear_nan(tr, R["position_x"].to_numpy(float) * MM_TO_M, grid)
    pos_y = interp_linear_nan(tr, R["position_y"].to_numpy(float) * MM_TO_M, grid)

    # velocidade da visao (mundo): mm/s->m/s (p/ sanity e IC)
    vel_x_w = interp_linear_nan(tr, R["velocity_x"].to_numpy(float) * MM_TO_M, grid)
    vel_y_w = interp_linear_nan(tr, R["velocity_y"].to_numpy(float) * MM_TO_M, grid)

    # interseccao temporal: descarta grade fora da cobertura da visao (NaN nas bordas)
    valid = ~(np.isnan(pos_x) | np.isnan(pos_y) | np.isnan(theta))
    if np.count_nonzero(valid) < 2:
        raise ValueError("sem sobreposicao temporal entre commands e processed_robots")
    keep = _contiguous_true_block(valid)
    grid = grid[keep]; cmd_vx = cmd_vx[keep]; cmd_vy = cmd_vy[keep]; cmd_w = cmd_w[keep]
    theta = theta[keep]; pos_x = pos_x[keep]; pos_y = pos_y[keep]
    vel_x_w = vel_x_w[keep]; vel_y_w = vel_y_w[keep]

    dt = np.empty_like(grid)
    dt[1:] = np.diff(grid)
    dt[0] = ts

    # serie de velocidade de CORPO derivada da POSICAO (estimador unico do loader);
    # da o velb0 de cada janela do rollout. velb0 do Segment = serie em t0=0.
    velb_x, velb_y, velb_w = body_velocity_series(grid, pos_x, pos_y, theta)
    pose0 = (float(pos_x[0]), float(pos_y[0]), float(theta[0]))
    velb0 = (float(velb_x[0]), float(velb_y[0]), float(velb_w[0]))

    md = dict(meta or {})
    md.setdefault("robot_id", rid)
    md.setdefault("team", team)
    md.setdefault("src_commands", str(commands_path))
    md.setdefault("src_robots", str(robots_path))
    md.setdefault("ts", ts)

    seg = Segment(meta=md, ts=ts, t=grid, dt=dt,
                  cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_w=cmd_w,
                  pos_x=pos_x, pos_y=pos_y, theta=theta,
                  vel_x_world=vel_x_w, vel_y_world=vel_y_w,
                  velb_x=velb_x, velb_y=velb_y, velb_w=velb_w,
                  pose0=pose0, velb0=velb0)
    if validate:
        validate_segment(seg, ts=ts)
    return seg


def load_segment(label: str, event: str, *, dataset: str = DATASET_DEFAULT,
                 data_root: Path | None = None, **kwargs) -> Segment:
    """Conveniencia: resolve caminhos a partir de (label, event, dataset)."""
    d = event_dir(label, event, dataset, data_root)
    meta = {"dataset": dataset, "label": label, "event": event}
    return load_event(d / "commands.parquet", d / "processed_robots.parquet",
                      meta=meta, **kwargs)


# ----------------------------------------------------------------------------
def _contiguous_true_block(valid: np.ndarray) -> np.ndarray:
    """Maior bloco contiguo de True (interp linear so produz NaN nas bordas)."""
    idx = np.flatnonzero(valid)
    keep = np.zeros_like(valid)
    keep[idx[0]:idx[-1] + 1] = True
    return keep


def body_velocity_series(t, pos_x, pos_y, theta):
    """Serie de velocidade de CORPO (vx_b, vy_b, w) POR AMOSTRA, derivada da
    POSICAO da visao via np.gradient e rotacionada mundo->corpo. Estimador UNICO
    do loader: o velb0 do Segment e o velb0 de cada janela do rollout vem daqui
    (serie[t0]). NAO usa a velocity_* logada. Em t=0 o np.gradient cai na diferenca
    forward (mesma convencao do velb0 antigo)."""
    if t.size < 2:
        z = np.zeros_like(t)
        return z, z, z.copy()
    vx_w = np.gradient(pos_x, t)
    vy_w = np.gradient(pos_y, t)
    w = np.gradient(theta, t)
    c, s = np.cos(theta), np.sin(theta)
    vx_b = c * vx_w + s * vy_w       # mundo -> corpo
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b, w


def segment_displacement_m(seg: Segment) -> float:
    """Maior distancia da pose truth ao ponto inicial (m) — proxy de 'movimento'."""
    dx = seg.pos_x - seg.pos_x[0]
    dy = seg.pos_y - seg.pos_y[0]
    return float(np.max(np.hypot(dx, dy)))


def segment_peak_cmd_speed(seg: Segment) -> float:
    """Pico da velocidade de comando translacional (m/s)."""
    return float(np.max(np.hypot(seg.cmd_vx, seg.cmd_vy)))


def segment_is_moving(seg: Segment, min_displacement_m: float = 0.1,
                      min_peak_cmd_speed: float = 0.0) -> bool:
    """Segmento 'com movimento' = deslocamento truth >= limiar (e, opcional, pico de
    comando >= limiar). Limiares configuraveis; descarta idle."""
    ok = segment_displacement_m(seg) >= min_displacement_m
    if min_peak_cmd_speed > 0.0:
        ok = ok and (segment_peak_cmd_speed(seg) >= min_peak_cmd_speed)
    return ok

"""Motor de rollout multi-horizonte em FREE-RUN.

DRIVER FINO: NAO implementa integracao, discretizacao nem dinamica. So janelamento,
condicao inicial via API da Plant, free-run e coleta. Toda a dinamica vem de
Plant.reset / Plant.prime_delay / Plant.step. NAO computa metrica.

Por janela (a cada stride):
  reset(pose0 = pose real em t0, velb0 = serie vel-de-corpo em t0)   [serie do loader]
  prime_delay(comandos imediatamente anteriores a t0; lookback generoso)
  free-run de H_max passos: pose_sim[0]=pose0; pose_sim[h]=step(cmd[t0+h-1], dt)
INVARIANTE: dentro da janela so se roda na propria predicao. Reinicializar por
janela e permitido; reancorar a cada passo (teacher forcing) e PROIBIDO — nenhum
indice de pose real > t0 e lido para conduzir a simulacao.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import load_twin_params
from .plant import Plant
from .schema import Segment

DEFAULT_HMAX = 90
DEFAULT_HORIZONS = (6, 12, 24, 42, 60, 90)
DEFAULT_STRIDE = 15
DEFAULT_LOOKBACK = 16
DEFAULT_TRANS_FLOOR = 0.05   # m/s
DEFAULT_ROT_FLOOR = 0.10     # rad/s


def max_command_delay(dt: float, params=None) -> int:
    """max sobre DOF de ceil(Td_dof/dt), lido de config/twin_params.json. Recalcula
    sozinho quando os params (ex.: de w) mudam. Casa com o tap do plant
    (floor(j - Td/dt) => ceil(Td/dt) amostras)."""
    p = params or load_twin_params()
    def m(dof: str) -> int:
        Td = p[dof].Td
        return 0 if Td <= 0.0 else int(math.ceil(Td / dt - 1e-9))
    return max(m("vx"), m("vy"), m("w"))


@dataclass
class Window:
    t0: int                    # indice da CI no segmento
    t0_time: float             # seg.t[t0] (s)
    pose_sim: np.ndarray       # (H_max+1, 3) free-run; linha h alinha com pose_real[h]
    pose_real: np.ndarray      # (H_max+1, 3) pose real em t0..t0+H_max
    cmd: np.ndarray            # (H_max, 3) comandos usados (t0..t0+H_max-1) — p/ estratificacao
    velb: np.ndarray           # (H_max+1, 3) serie vel-de-corpo na janela — p/ estratificacao
    velb0: tuple               # CI de velocidade de corpo (= serie em t0)
    trans_peak: float          # m/s   (conteudo de comando da janela)
    trans_mean: float          # m/s
    rot_peak: float            # rad/s
    rot_mean: float            # rad/s
    idle: bool                 # quase-parada (sinalizada, NAO apagada)


@dataclass
class WindowRollout:
    meta: dict
    windows: list

    @property
    def n_total(self) -> int:
        return len(self.windows)

    @property
    def n_idle(self) -> int:
        return sum(1 for w in self.windows if w.idle)


# =========================================================================
def free_run(plant: Plant, win_cmds: np.ndarray, pose0, velb0, dt: float,
             prev_cmds=None) -> np.ndarray:
    """Conduz a planta em free-run por len(win_cmds) passos e devolve pose_sim
    (H+1, 3) com pose_sim[0]=pose0. Driver FINO: reset -> (prime) -> step loop.
    Nenhuma dinamica/integracao aqui — tudo vem da Plant."""
    plant.reset(pose0=pose0, velb0=velb0)
    if prev_cmds is not None and len(prev_cmds) > 0:
        plant.prime_delay(prev_cmds)
    h_max = len(win_cmds)
    pose_sim = np.empty((h_max + 1, 3), dtype=float)
    pose_sim[0] = pose0
    for h in range(1, h_max + 1):
        pose_sim[h] = plant.step(win_cmds[h - 1], dt)
    return pose_sim


def rollout_segment(seg: Segment, plant: Plant, *, h_max: int = DEFAULT_HMAX,
                    horizons=DEFAULT_HORIZONS, stride: int = DEFAULT_STRIDE,
                    lookback: int = DEFAULT_LOOKBACK,
                    trans_floor: float = DEFAULT_TRANS_FLOOR,
                    rot_floor: float = DEFAULT_ROT_FLOOR, params=None) -> WindowRollout:
    """Janela o segmento e roda free-run por janela. Veja o invariante no topo."""
    if max(horizons) > h_max:
        raise ValueError(f"horizontes {horizons} excedem H_max={h_max}")
    dt = seg.ts
    N = seg.n
    md = max_command_delay(dt, params)
    idx_min = md + 1                       # precisa de comandos anteriores p/ primar

    cmd = np.column_stack([seg.cmd_vx, seg.cmd_vy, seg.cmd_w])
    velb = np.column_stack([seg.velb_x, seg.velb_y, seg.velb_w])   # serie do loader
    pos = np.column_stack([seg.pos_x, seg.pos_y, seg.theta])

    windows: list[Window] = []
    for t0 in range(idx_min, N - h_max, stride):
        pose0 = (float(pos[t0, 0]), float(pos[t0, 1]), float(pos[t0, 2]))   # pose real em t0
        velb0 = (float(velb[t0, 0]), float(velb[t0, 1]), float(velb[t0, 2]))  # serie em t0
        P = min(t0, lookback)
        prev = cmd[t0 - P:t0]              # comandos antes de t0 (antigo->recente)
        win_cmds = cmd[t0:t0 + h_max]      # cmd[t0 .. t0+H_max-1]

        pose_sim = free_run(plant, win_cmds, pose0, velb0, dt, prev_cmds=prev)
        pose_real = pos[t0:t0 + h_max + 1].copy()   # coleta (NAO usada p/ conduzir)
        velb_win = velb[t0:t0 + h_max + 1].copy()

        tr = np.hypot(win_cmds[:, 0], win_cmds[:, 1])
        rot = np.abs(win_cmds[:, 2])
        tp, tm = float(tr.max()), float(tr.mean())
        rp, rm = float(rot.max()), float(rot.mean())
        idle = (tp < trans_floor) and (rp < rot_floor)

        windows.append(Window(t0=t0, t0_time=float(seg.t[t0]), pose_sim=pose_sim,
                              pose_real=pose_real, cmd=win_cmds.copy(), velb=velb_win,
                              velb0=velb0, trans_peak=tp, trans_mean=tm,
                              rot_peak=rp, rot_mean=rm, idle=idle))

    meta = dict(event=seg.meta.get("event"), label=seg.meta.get("label"),
                dataset=seg.meta.get("dataset"), ts=dt, n_samples=N, h_max=h_max,
                horizons=tuple(horizons), stride=stride, lookback=lookback,
                max_delay=md, idx_min=idx_min, plant=type(plant).__name__,
                trans_floor=trans_floor, rot_floor=rot_floor)
    return WindowRollout(meta=meta, windows=windows)

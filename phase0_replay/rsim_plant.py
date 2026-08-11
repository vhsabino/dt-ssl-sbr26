"""RSimReplay(Plant): replay free-run sobre o rSim (rSoccer / rc-robosim, ODE).

Espelha a interface Plant (reset/prime_delay/step) para entrar no MESMO motor de
rollout (rollout.free_run) e na MESMA metrica (metrics.evaluate) que o twin
(AnalyticFOPDTPlant) e o IdealKinematicPlant. Toda a dinamica vem do engine; aqui
so ha fiacao: grade temporal, frame de comando e justica de condicao inicial.

Decisoes (validadas por calibracao headless do rc-robosim 1.2, cp310/ODE):
  - ACAO = (vx, vy, w) de CORPO, aplicada DIRETO ao robosim.SSL.step com
    wheel_speed=False (slot 0 = 0; slots 1,2,3 = vx,vy,w). Sem controle de
    movimento. Casa com o log; NAO se rotaciona nada.
      * vx em m/s (ganho ~1.0), vy em m/s (ganho lateral ~0.52 — propriedade do
        rSim SSL), w em RAD/S (ganho ~0.95). Comando angular NAO e convertido.
  - POSE via get_state(): x,y em m; theta em GRAUS -> convertido p/ rad e
    unwrapped continuamente. (A metrica usa wrap_to_pi, entao offsets de 2*pi sao
    inocuos.)
  - dt interno do engine e INTEIRO em ms. Para casar a grade 1/60 (=16.6667 ms,
    nao 16 ms), roda-se o engine em SUBSTEP pequeno (sub_ms, default 1 ms)
    ACUMULANDO ~16.667 ms por passo 1/60 e LE-SE a pose na grade por interpolacao
    LINEAR (theta via unwrap), igual ao loader.
"""
from __future__ import annotations

import math

import numpy as np

import robosim

from .plant import Plant, wrap_to_pi

# rSoccer SSL: field_type=2 (Div B). 1 robo azul, 0 amarelos. Comando 8-wide.
_FIELD_TYPE = 2
_DT_GRID = 1.0 / 60.0


class RSimReplay(Plant):
    def __init__(self, *, field_type: int = _FIELD_TYPE, sub_ms: int = 1,
                 warmup: bool = True):
        if sub_ms <= 0:
            raise ValueError("sub_ms deve ser >= 1 (ms)")
        self.field_type = int(field_type)
        self.sub_dt = float(sub_ms) / 1000.0
        self.warmup = bool(warmup)
        # engine criado UMA vez; reposicionado por reset (reuso evita custo de spawn).
        self._sim = robosim.SSL(self.field_type, 1, 0, int(sub_ms),
                                [0.0, 0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]], [])
        self._cmd_buf = np.zeros((1, 8), dtype=np.float64)

    # ---- engine helpers ---------------------------------------------------
    def _raw_pose(self):
        st = self._sim.get_state()
        return float(st[5]), float(st[6]), math.radians(float(st[7]))  # x,y(m), th(rad)

    def _set_cmd(self, cmd):
        c = self._cmd_buf
        c[0, 0] = 0.0                 # wheel_speed = False -> velocidades de corpo
        c[0, 1] = float(cmd[0])       # vx (m/s)
        c[0, 2] = float(cmd[1])       # vy (m/s)
        c[0, 3] = float(cmd[2])       # w  (rad/s)
        return c

    def _grid_step(self, cmd):
        """Avanca o engine por _DT_GRID sob `cmd`, em substeps de sub_dt, e devolve
        a pose BRUTA (x,y,th_unwrapped) interpolada LINEARMENTE na grade 1/60."""
        c = self._set_cmd(cmd)
        self._target += _DT_GRID
        # mantem so a amostra anterior e a atual que cercam o alvo (memoria O(1)).
        prev_t, (prev_x, prev_y, prev_th) = self._sim_t, self._cur
        while self._sim_t < self._target - 1e-12:
            self._sim.step(c)
            self._sim_t += self.sub_dt
            rx, ry, rth = self._raw_pose()
            rth = self._last_th + wrap_to_pi(rth - self._last_th)   # unwrap continuo
            self._last_th = rth
            prev_t, (prev_x, prev_y, prev_th) = self._cur_t, self._cur
            self._cur_t, self._cur = self._sim_t, (rx, ry, rth)
        # interp linear no alvo entre (prev) e (cur)
        t0, t1 = prev_t, self._cur_t
        x1, y1, th1 = self._cur
        a = 0.0 if t1 <= t0 else (self._target - t0) / (t1 - t0)
        x = prev_x + a * (x1 - prev_x)
        y = prev_y + a * (y1 - prev_y)
        th = prev_th + a * (th1 - prev_th)
        return x, y, th

    def _anchor_to(self, raw):
        """Fixa a re-ancora rigida SE(2) tal que T(raw_at_t0) == pose0."""
        xa, ya, tha = raw
        self._ax, self._ay, self._ath = xa, ya, tha
        self._rot = wrap_to_pi(self._th0 - tha)
        self._crot, self._srot = math.cos(self._rot), math.sin(self._rot)

    def _apply_anchor(self, raw):
        xb, yb, thb = raw
        dx, dy = xb - self._ax, yb - self._ay
        x = self._x0 + self._crot * dx - self._srot * dy
        y = self._y0 + self._srot * dx + self._crot * dy
        th = thb + self._rot
        return x, y, th

    # ---- Plant API --------------------------------------------------------
    def reset(self, pose0, velb0=(0.0, 0.0, 0.0)) -> None:
        self._x0, self._y0, self._th0 = (float(v) for v in pose0)
        # engine sempre parte de uma origem bruta fixa; a re-ancora cuida do resto
        # (velb0 NAO e setavel no rSim -> ignorado de proposito; ver warm-up).
        self._sim.reset([0.0, 0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]], [])
        self._sim_t = 0.0
        self._target = 0.0
        rx, ry, rth = self._raw_pose()
        self._last_th = rth
        self._cur_t, self._cur = 0.0, (rx, ry, rth)
        # ancora default (sem warm-up): mapeia a origem bruta -> pose0
        self._anchor_to((rx, ry, rth))
        # estado exposto pela interface Plant
        self.x, self.y, self.th = self._x0, self._y0, self._th0
        self._vbody = np.asarray(velb0, dtype=float).reshape(3)

    def prime_delay(self, prev_cmds) -> None:
        """Warm-up (opcao A). NAO e atraso de transporte (o rSim nao o modela)."""
        if not self.warmup or prev_cmds is None or len(prev_cmds) == 0:
            return
        raw = (self._ax, self._ay, self._ath)
        for c in prev_cmds:
            raw = self._grid_step(c)        # acelera de ~repouso seguindo o historico
        self._anchor_to(raw)                # A := pose bruta em t0 -> pose0
        # nao mexe em self.x/y/th: free_run forca pose_sim[0]=pose0

    def step(self, cmd, dt: float):
        raw = self._grid_step(cmd)
        self.x, self.y, self.th = self._apply_anchor(raw)
        self._vbody = np.asarray(cmd, dtype=float).reshape(3)   # proxy (engine nao expoe v)
        return (self.x, self.y, self.th)

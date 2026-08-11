"""Primitivas de reamostragem que REPRODUZEM o pipeline de ID em MATLAB.

A etapa de reamostragem do gemeo (matlab/sysid/build_idData_per_dof.m) usa, para
cada evento:

    tgrid  = (tc(1):Ts:tc(end))'                       % colon do MATLAB, Ts=1/60
    cmd_*  = interp1(tc, move_*, tgrid, 'previous', 0)  % ZOH (segura amostra anterior)
    pose_* = interp1(tr, pose_*, tgrid, 'linear', NaN)  % linear, NaN fora do dominio
    theta  = interp1(tr, unwrap(position_w), tgrid, 'linear', NaN)

Este modulo re-implementa as MESMAS tres operacoes em numpy. A equivalencia
numerica e checada por teste cross-linguagem contra uma grade exportada do MATLAB
(automation/export_resample_reference.m -> tests/fixtures/*.parquet).
"""
from __future__ import annotations

import numpy as np


def matlab_colon(a: float, b: float, step: float) -> np.ndarray:
    """Reproduz o operador `a:step:b` do MATLAB para passo em ponto flutuante.

    MATLAB inclui o ultimo ponto se `a + n*step` cair dentro de uma tolerancia de
    `b`. Replicamos isso com tolerancia relativa em `(b-a)/step` (para nao perder o
    ultimo ponto por erro de float) e poda de qualquer overshoot acima de `b+tol`.
    """
    if step <= 0:
        raise ValueError("step deve ser > 0")
    ndiv = (b - a) / step
    n = int(np.floor(ndiv + 1e-9))            # numero de intervalos (tol p/ float)
    if n < 0:
        return np.empty(0, dtype=float)
    t = a + step * np.arange(n + 1, dtype=float)
    tol = 4.0 * np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    while t.size > 0 and t[-1] > b + tol:     # poda overshoot (como o MATLAB)
        t = t[:-1]
    return t


def interp_previous(x: np.ndarray, v: np.ndarray, xq: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Equivale a interp1(x, v, xq, 'previous', fill).

    Para xq dentro de [x[0], x[-1]] devolve o valor da amostra ANTERIOR (<= xq);
    fora do dominio (xq < x[0] OU xq > x[-1]) devolve `fill` (o extrapval escalar
    do MATLAB vale para os dois lados). Requer x ordenado e unico.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    xq = np.asarray(xq, dtype=float)
    idx = np.searchsorted(x, xq, side="right") - 1     # ultimo x <= xq
    out = np.full(xq.shape, float(fill), dtype=float)
    inside = (idx >= 0) & (xq <= x[-1])
    out[inside] = v[idx[inside]]
    return out


def interp_linear_nan(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Equivale a interp1(x, v, xq, 'linear', NaN).

    Linear dentro de [x[0], x[-1]]; NaN fora. (np.interp satura nas bordas, entao
    mascaramos explicitamente o que cai fora do dominio.) Requer x ordenado/unico.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    xq = np.asarray(xq, dtype=float)
    out = np.interp(xq, x, v)
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out

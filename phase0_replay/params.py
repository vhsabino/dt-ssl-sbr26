"""Parametros do gemeo (K, tau, Td por DOF) lidos de config/twin_params.json.

O JSON e extraido UMA vez de models/sysid_vxvy_v2.mat (modelo promovido):
vx/vy identificados; w e PLACEHOLDER (marcado). Ordem fixa do SLDD em 'order'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .io import repo_root

DOFS = ("vx", "vy", "w")


@dataclass(frozen=True)
class DofParam:
    K: float
    tau: float
    Td: float
    placeholder: bool = False


def twin_params_path() -> Path:
    return repo_root() / "config" / "twin_params.json"


def load_twin_params(path: Path | None = None) -> dict[str, DofParam]:
    """Le config/twin_params.json -> {'vx':DofParam, 'vy':DofParam, 'w':DofParam}."""
    p = path or twin_params_path()
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    dof = cfg["dof"]
    return {d: DofParam(K=float(dof[d]["K"]), tau=float(dof[d]["tau"]),
                        Td=float(dof[d]["Td"]), placeholder=bool(dof[d].get("placeholder", False)))
            for d in DOFS}

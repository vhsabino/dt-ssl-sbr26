"""Hold-out: fonte unica em config/holdout_events.json (IDs literais dos scripts
de avaliacao MATLAB; NAO regenerado via rng). Seleciona so hold-out e, dentre
eles, so segmentos com movimento (descarta idle)."""
from __future__ import annotations

import json
from pathlib import Path

from .io import repo_root, load_segment, segment_is_moving
from .schema import Segment, TS_DEFAULT


def holdout_config_path(data_root_repo: Path | None = None) -> Path:
    return (data_root_repo or repo_root()) / "config" / "holdout_events.json"


def load_holdout_list(path: Path | None = None) -> dict:
    """Le config/holdout_events.json -> dict com dataset, team, events[{label,event}]."""
    p = path or holdout_config_path()
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["events"] = [e for e in cfg["events"]]   # lista de {label,event}
    return cfg


def load_holdout_segments(*, ts: float = TS_DEFAULT, min_displacement_m: float = 0.1,
                          min_peak_cmd_speed: float = 0.0, data_root: Path | None = None,
                          drop_idle: bool = True, config: Path | None = None
                          ) -> tuple[list[Segment], list[dict]]:
    """Carrega os Segments do hold-out (so com movimento, se drop_idle).

    Retorna (segments, dropped) onde dropped lista {label,event,reason}.
    """
    cfg = load_holdout_list(config)
    dataset = cfg.get("dataset")
    team = cfg.get("team", "allies")
    segments: list[Segment] = []
    dropped: list[dict] = []
    for e in cfg["events"]:
        seg = load_segment(e["label"], e["event"], dataset=dataset, team=team,
                           ts=ts, data_root=data_root)
        if drop_idle and not segment_is_moving(seg, min_displacement_m, min_peak_cmd_speed):
            dropped.append({"label": e["label"], "event": e["event"], "reason": "idle"})
            continue
        segments.append(seg)
    return segments, dropped

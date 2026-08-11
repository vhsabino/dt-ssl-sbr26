"""phase0_replay — harness de replay do gemeo digital SSL (FOPDT por DOF).

Modulo de carregamento (loader). Le os dois parquets assincronos/irregulares de
um evento (commands + processed_robots), converte para SI estrito, reamostra
AMBOS para a grade uniforme Ts = 1/60 s (reproduzindo a etapa de reamostragem do
pipeline de identificacao em MATLAB, build_idData_per_dof) e devolve um Segment
canonico.
"""
from .schema import Segment, TS_DEFAULT
from .io import (load_event, load_segment, repo_root, event_dir,
                 body_velocity_series)
from .splits import load_holdout_list, load_holdout_segments
from .resample import matlab_colon, interp_previous, interp_linear_nan
from .params import DofParam, load_twin_params
from .plant import Plant, AnalyticFOPDTPlant, IdealKinematicPlant, wrap_to_pi
from .rollout import (Window, WindowRollout, rollout_segment, free_run,
                      max_command_delay, DEFAULT_HMAX, DEFAULT_HORIZONS,
                      DEFAULT_STRIDE)
from .metrics import (evaluate, build_error_table, classify_bin,
                      bin_distribution, format_bin_distribution,
                      load_metrics_config, make_figures, markdown_report,
                      rank_biserial, ErrorTable)

__all__ = [
    "Segment", "TS_DEFAULT",
    "load_event", "load_segment", "repo_root", "event_dir", "body_velocity_series",
    "load_holdout_list", "load_holdout_segments",
    "matlab_colon", "interp_previous", "interp_linear_nan",
    "DofParam", "load_twin_params",
    "Plant", "AnalyticFOPDTPlant", "IdealKinematicPlant", "wrap_to_pi",
    "Window", "WindowRollout", "rollout_segment", "free_run", "max_command_delay",
    "DEFAULT_HMAX", "DEFAULT_HORIZONS", "DEFAULT_STRIDE",
    "evaluate", "build_error_table", "classify_bin", "bin_distribution",
    "format_bin_distribution", "load_metrics_config", "make_figures",
    "markdown_report", "rank_biserial", "ErrorTable",
]

# A Behavioral Model for Digital Twins of Omnidirectional Soccer Robots: Diagnosing Identifiability from Competition Logs

Reproduction package. Every number in the paper is produced by the code here
from the configurations in `config/`, and each claim below names the exact
script and artefact that backs it.

> **Anonymised for double-blind review.** No author, institution or team name
> appears in this repository or in its commit history.

## Layout

| Path | Content |
|---|---|
| `phase0_replay/` | Python evaluation harness: plants, free-run rollout, multi-horizon metric, statistical gate, rSim wrapper |
| `matlab/` | MATLAB identification: per-DOF FOPDT fits, excitation diagnostics, SE(2) promotion gate |
| `scripts/` | Every analysis and figure in the paper, one script per claim |
| `config/` | Identified parameters, metric configuration, frozen hold-out list, robot geometry |
| `docs/` | Pre-registration document, committed before the yaw estimation |
| `results/` | The exact artefacts the paper cites |
| `figures/` | Figures as they appear in the paper, plus the figure specification |

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. MATLAB R2026a with the System Identification Toolbox is needed
only for `matlab/`; the evaluation harness and every analysis below are pure
Python. `robosim` (the ODE engine behind rSim) is Linux-only.

## Data

Competition logs and the dedicated rotation collection. Place
the decoded splits under

```
data/extracted/<log>/splits/<label>/<event>/{commands,processed_robots,telemetry}.parquet
```

`config/holdout_events.json` is the single source of truth for the ten hold-out
events. It is a literal list, not a seeded draw, so the split cannot drift. The
five events that validated `v_x` and `v_y` are a subset of it, and the other
five carry a label that entered neither fit: no event contributed to both
parameter selection and fidelity scoring.

## One command per claim

| Claim in the paper | Script | Artefact |
|---|---|---|
| Per-DOF FOPDT parameters, hold-out fit (Table I) | `matlab -batch "identify_trans_dof('x'); identify_trans_dof('y')"`, `identify_omega_rotate` | — |
| Position and heading error, three plants, six horizons (Table II, Fig. 2a) | `scripts/epos_eang_three_models.py` | `results/epos_eang_three_models_*.md` |
| Per-event statistics, 10/10 events, 73.5 s (Fig. 2b) | `scripts/clustered_stats.py` | `results/clustered_stats_*.md` |
| Order-independent ablation (Shapley) and initial-condition arms | `scripts/ablation_orderfree.py` | `results/ablation_orderfree_*.md` |
| Robustness to a strictly causal initial velocity | `scripts/causal_ic_check.py` | `results/causal_ic_check_*.md` |
| Screening: actuation share and explained variance per DOF | `scripts/excitation_per_dof.py` | `results/excitation_per_dof_*.md` |
| Dilution analysis, threshold sensitivity, wheel-angle sweep | `scripts/analyze_omega_excitation.py` | `results/omega_excitation_analysis_*.json` |
| Closed-loop yaw fit under the paper's hold-out criterion | `scripts/r3_closedloop_omega_fit.py` | `results/r3_closedloop_omega_fit_*.md` |
| Negative control (joint Bayesian calibration), per event | `scripts/negcontrol_perevent.py` | `results/negcontrol_perevent_*.md` |
| Matched-delay ablation on rSim | `scripts/eval_rsim_delay_ablation.py --cache` then `--finalize` | `results/eval_rsim_delay_ablation_*.md` |
| Main figure | `scripts/fig_main_result.py` | `figures/main_result_en.png` |
| Yaw step response | `scripts/fig_omega_stepresponse_en.py` | `figures/omega_stepresponse_en.png` |

rSim rollouts are cached in `results/_cache_rsim_rollouts.pkl` so the ablation
does not re-simulate.

## Definitions used in the paper

**hold-out fit.** Normalised RMSE as a percentage, of the model output against
the measured body velocity, on events never used for fitting:
`fit = 100 · (1 − ‖y − ŷ‖ / ‖y − mean(y)‖)`. 100% is exact, 0% is no better than
predicting the mean. Computed per event, then averaged over hold-out events. The
model runs in simulation, driven by commands only, not one-step prediction.

**free run.** The model starts at a known pose and is driven only by the
recorded command stream, with no correction from measured data inside the
window. Scoring one step at a time instead would measure sensor noise and
overstate fidelity.

**actuation share.** The fraction of the commanded per-wheel effort attributable
to one command channel. Computed from the log alone, before any fit.

**explained variance.** The fraction of the attained body velocity that its own
command accounts for, once aligned by the identified dead time.

**placeholder.** The no-dynamics reference used by the yaw promotion gate:
`K = 1`, `τ = 0.1 s`, `Td = 0`.

## A note on the initial condition

The loader estimates the initial body velocity with a central difference, which
at the window anchor reads one sample from inside the evaluation window.
`scripts/causal_ic_check.py` re-runs everything with a strictly causal backward
difference: both plants get worse, the gap widens, and the verdict is unchanged.
The result does not depend on that sample.

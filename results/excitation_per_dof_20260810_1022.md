# Excitation screening per DOF, competition logs — 20260810_1022

Each command is aligned by its identified dead time; the measured body velocity
uses the same pre-processing as the paper (Savitzky–Golay, window 7, order 2,
followed by central differences).

| DOF | label | n | corr(cmd, meas) | R² (explained variance) |
|---|---|---|---|---|
| vx | `front_to_back` | 1964 | **0.979** | **0.959** |
| vy | `side_to_side` | 7228 | **0.968** | **0.936** |
| ω  | `shoot_to_goal` | 7484 | **0.746** | **0.557** |

Both statistics are scale-invariant, which matters here: vision position is
stored in millimetres while commands are in SI, so absolute standard deviations
of the two signals are not directly comparable and are omitted.

Reading: for the translational channels the command explains most of the
variance of the attained velocity; for yaw it does not. Nearly half of the
rotation the vision system measures in competition is not attributable to the
yaw command. This is the pre-fit half of the screening procedure — the other
half is the actuation share, in `omega_excitation_analysis_*.json`.

# Order-independent ablation and initial-condition arms — 20260810_1021

Median position error (m) in the translation-dominated bin, h_ref = 0.4 s, same
windows throughout.

## 1. Full 2^3 factorial (all eight combinations, measured initial condition)

| K | tau | Td | e_pos (m) |
|---|---|---|---|
| off | off | off | 0.1005 |
| off | off | on | 0.0569 |
| off | on | off | 0.0509 |
| off | on | on | 0.0467 |
| on | off | off | 0.1093 |
| on | off | on | 0.0544 |
| on | on | off | 0.0455 |
| on | on | on | 0.0388 |

## 2. Shapley attribution (order-INDEPENDENT)

Mean reduction in e_pos attributable to each group, averaged over all orderings:

| group | contribution (m) | share of the improvement |
|---|---|---|
| gain K | +0.0010 | +2% |
| time constant tau | +0.0341 | +55% |
| dead time Td | +0.0266 | +43% |

Sum = +0.0617 m = total improvement (ideal 0.1005 → full model 0.0388).

A cumulative ablation that enables tau before Td assigns the lag about nine
tenths of the credit. That is an artefact of ordering: the two terms are partly
redundant, since both postpone the response. Averaged over orderings they are
comparable contributors.

## 3. How much of the advantage is the initial condition, not the dynamics

| plant | initial condition | e_pos (m) |
|---|---|---|
| ideal kinematics (memoryless) | irrelevant | 0.1005 |
| **constant-velocity extrapolation** | **uses ONLY the measured IC** | **0.1600** |
| full FOPDT | starts from rest (as rSim does) | 0.1137 |
| full FOPDT | measured IC | 0.0388 |

- Constant-velocity extrapolation, which uses the initial condition and ignores
  the commands entirely, gives 0.1600 m — worse than ideal kinematics.
- The model started from rest gives 0.1137 m, also no better than ideal
  kinematics.
- Only the combination reaches 0.0388 m. The advantage is therefore a joint
  effect: the identified dynamics are what make the measured initial state
  usable, and neither ingredient helps on its own.

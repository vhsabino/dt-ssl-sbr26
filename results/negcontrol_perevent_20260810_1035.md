# Negative control (joint Bayesian calibration), event-clustered — 20260810_1035

Same protocol as the main result: per-event medians, n = 10, h_ref = 0.4 s,
translation-dominated bin. This supersedes the earlier window-level figures,
which were not commensurable with the headline analysis.

| comparison | median A | median B | median gap | A wins | Wilcoxon p | rank-biserial |
|---|---|---|---|---|---|---|
| per-DOF vs ideal | 0.0421 | 0.1177 | +0.0689 | 10/10 | 0.0009766 | +1.000 |
| bayesopt vs ideal | 0.0421 | 0.1177 | +0.0690 | 10/10 | 0.0009766 | +1.000 |
| per-DOF vs bayesopt | 0.0421 | 0.0421 | +0.0038 | 7/10 | 0.04199 | +0.400 |

## Reading

- per-DOF vs ideal: gap +0.0689 m, 10/10, p = 0.000977 — the main result, robust.
- bayesopt vs ideal: gap +0.0690 m, 10/10, p = 0.000977 — indistinguishable from
  the per-DOF model against the same baseline.
- **per-DOF vs bayesopt: gap +0.0038 m, 7 of 10 events, p = 0.042, effect
  +0.400.** Four millimetres. We read this as a tie, not a win.

The conclusion that survives is that four hundred evaluations of joint
9-parameter calibration on these logs buy nothing over identifying each DOF
separately — which is what the excitation argument predicts.

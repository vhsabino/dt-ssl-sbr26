# Closed-loop yaw fit under the paper's hold-out criterion — 20260810_1036

Same criterion as the open-loop row (Eq. 2 in the paper): event-wise split, mean
over the hold-out events. This closes the gap that left the closed-loop attempt
without a number comparable to the dedicated collection.

- `shoot_to_goal` events used: 18 (12 training / 6 hold-out)
- Parameters: K = 1.3775, tau = 0.0430 s, Td = 0.1117 s
- **hold-out fit = 44.8%** (per event: 26.4, 46.8, 45.7, 60.3, 51.4, 38.0)
- training fit (median) = 52.6%

Direct comparison, now under the SAME criterion:

| regime | K | tau (s) | Td (s) | hold-out fit |
|---|---|---|---|---|
| closed-loop (competition) | 1.378 | 0.0430 | 0.1117 | **44.8%** |
| open-loop (dedicated) | 1.002 | 0.0828 | 0.0890 | **71.1%** |

Pre-registered gate: hold-out fit ≥ 60%. Only the dedicated collection clears
it. Note that the earlier campaign reported 52.6% for the closed-loop attempt;
that figure is the *training* median under this recomputation, and both values
sit below the gate, so the conclusion is unchanged.

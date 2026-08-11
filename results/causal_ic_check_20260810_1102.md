# Causal vs central initial-velocity estimator — 20260810_1102

The evaluation harness (`phase0_replay/io.py:body_velocity_series`) estimates the
initial body velocity with `np.gradient`, a CENTRAL difference, which at the
window anchor reads sample t0+1 — that is, ONE sample (16.7 ms) from inside the
evaluation window. (The Savitzky–Golay filter sits in the IDENTIFICATION path,
not in the evaluation path.) Here the initial velocity is recomputed with a
strictly causal backward difference and everything is re-scored.

| initial-velocity estimator | e_pos h=0.1 | h=0.4 | h=1.5 | per-event gap | wins | p |
|---|---|---|---|---|---|---|
| central (`np.gradient`, ±1 sample) | 0.0079 | 0.0388 | 0.1193 | +0.0689 | 10/10 | 0.0009766 |
| causal (backward difference, past only) | 0.0128 | 0.0468 | 0.1335 | +0.0706 | 10/10 | 0.0009766 |

Difference at h_ref: +0.0080 m (+20.6%).

Conclusion: the verdict does not depend on the estimator. Both plants get worse
under a strictly causal initial velocity, but the gap in favour of the model
*widens*, and it still wins in all ten hold-out events at the same p. This
closes the objection that part of the advantage came from seeing inside the
evaluation window.

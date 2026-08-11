# Event-clustered statistics — 20260809_2309

Unit of analysis = hold-out event. n = **10 events** (against n = 49 windows in
the window-level analysis).

Dense translation-dominated windows: 264; non-overlapping: 49; scored motion =
49 × 1.5 s = **73.5 s**.

| event (idx) | model e_pos (m) | ideal e_pos (m) | gap (m) | windows |
|---|---|---|---|---|
| 0 | 0.0172 | 0.0547 | +0.0375 | 27 |
| 1 | 0.0581 | 0.1618 | +0.1037 | 33 |
| 2 | 0.0449 | 0.1140 | +0.0692 | 33 |
| 3 | 0.0224 | 0.0530 | +0.0306 | 33 |
| 4 | 0.0213 | 0.0530 | +0.0317 | 28 |
| 5 | 0.0552 | 0.1214 | +0.0662 | 17 |
| 6 | 0.0393 | 0.1079 | +0.0686 | 13 |
| 7 | 0.0436 | 0.1290 | +0.0854 | 27 |
| 8 | 0.0540 | 0.1619 | +0.1079 | 31 |
| 9 | 0.0406 | 0.1376 | +0.0969 | 22 |

- Paired Wilcoxon **per event** (one-sided, model < ideal): **p = 0.0009766**,
  which is the smallest value attainable at n = 10.
- Median per-event gap: **+0.0689 m**; bootstrap 95% CI over events
  [+0.0375, +0.0969].
- Paired rank-biserial per event: **+1.000** (10 in favour, 0 against).

All at h_ref = 0.4 s, translation-dominated bin, model vs ideal kinematics.
Windows from one event share robot, surface, session and controller, so the
event is the unit that the data structure supports.

# Leave-one-event-out per translational DOF — 20260810_1743

Full refit on every fold, keeping the P1D structure and the fit metric of
Eq. (2). This does NOT alter the promoted model in `config/twin_params.json`: it
is a dispersion analysis, not a re-identification. It answers the recurring
objection that the `vx` fit rests on a single hold-out event.

## vx (`front_to_back`, 4 events, 4 folds)

| held-out event | fit (%) | K | tau (s) | Td (s) |
|---|---|---|---|---|
| front_to_back_01_01 | 85.8 | 1.126 | 0.0933 | 0.0563 |
| front_to_back_01_02 | 82.5 | 1.128 | 0.0948 | 0.0569 |
| front_to_back_01_03 | 86.2 | 1.130 | 0.0881 | 0.0556 |
| front_to_back_01_04 | 82.7 | 1.134 | 0.0885 | 0.0574 |

- mean fit **84.3%**, standard deviation **2.0 p.p.**, range 82.5–86.2%
- mean K 1.129, standard deviation 0.003 (promoted value: 1.119)

## vy (`side_to_side`, 13 events, 13 folds)

| held-out event | fit (%) | K | tau (s) | Td (s) |
|---|---|---|---|---|
| side_to_side_01_01 | 79.1 | 1.069 | 0.1435 | 0.0419 |
| side_to_side_01_02 | 82.6 | 1.069 | 0.1416 | 0.0450 |
| side_to_side_01_03 | 82.7 | 1.069 | 0.1421 | 0.0468 |
| side_to_side_01_04 | 81.6 | 1.068 | 0.1408 | 0.0471 |
| side_to_side_01_05 | 83.3 | 1.070 | 0.1409 | 0.0461 |
| side_to_side_01_06 | 77.2 | 1.064 | 0.1456 | 0.0470 |
| side_to_side_01_07 | 83.1 | 1.064 | 0.1450 | 0.0483 |
| side_to_side_01_08 | 88.5 | 1.065 | 0.1450 | 0.0481 |
| side_to_side_02_01 | 82.0 | 1.065 | 0.1462 | 0.0476 |
| side_to_side_02_02 | 84.3 | 1.065 | 0.1455 | 0.0473 |
| side_to_side_02_03 | 83.1 | 1.064 | 0.1456 | 0.0469 |
| side_to_side_02_04 | 82.9 | 1.065 | 0.1460 | 0.0468 |
| side_to_side_02_05 | 85.4 | 1.065 | 0.1451 | 0.0482 |

- mean fit **82.8%**, standard deviation **2.7 p.p.**, range 77.2–88.5%
- mean K 1.066, standard deviation 0.002 (promoted value: 1.052)

## Reading

The refitted gain stays within 0.015 of the promoted value on both channels, and
the fold-to-fold spread of the fit is a few percentage points. The single
hold-out event used for `vx` in the main identification is therefore not an
outlier, and the promoted parameters are stable across which event is withheld.

These figures come from an independent Python reimplementation of the P1D fit,
not from the MATLAB pipeline that produced the promoted model, which is why they
sit slightly below the 86.7% and 87.1% reported in the paper.

# twin (FOPDT) vs rSim (ODE replay) — metrica SE(2) multi-horizonte

Modo de justica de IC: **A (warm-up Lw=15 + re-anchor rigida SE(2))**. Lw(warm-up)=15 amostras, sub_ms=1, stride=15, h_max=90.
Config metrica: horizontes(amostras)=[np.int64(6), np.int64(12), np.int64(24), np.int64(42), np.int64(60), np.int64(90)] (s=[0.1, 0.2, 0.4, 0.7, 1.0, 1.5]), h_ref=24, N_min=10, alpha=0.05, effect_min=0.2, bloco_MBB=6 windows.

## Distribuicao de bins

| bin | total (dense) | independent |
|---|---:|---:|
| translation-dominated | 264 | 49 |
| mixed | 0 | 0 |
| rotation-dominated | 1 | 0 |
| **TODOS** | 265 | 49 |

## TODOS (agregado)
N total=265, independent=49

| h (s) | twin e_pos | rSim (ODE replay) e_pos | gap (m) | gap CI95 | p (non-overl.) | effect |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0076 | 0.0572 | +0.0496 | [+0.0451,+0.0525] | 1.56e-13 | +0.979 |
| 0.20 | 0.0177 | 0.0938 | +0.0761 | [+0.0669,+0.0812] | 1.24e-13 | +0.980 |
| 0.40 | 0.0383 | 0.1315 | +0.0932 | [+0.0792,+0.1026] | 4.38e-11 | +0.922 |
| 0.70 | 0.0637 | 0.1731 | +0.1095 | [+0.0871,+0.1336] | 8.18e-09 | +0.838 |
| 1.00 | 0.0908 | 0.2141 | +0.1233 | [+0.1038,+0.1437] | 1.31e-09 | +0.871 |
| 1.50 | 0.1228 | 0.2924 | +0.1696 | [+0.1198,+0.2109] | 1.03e-06 | +0.731 |

## translation-dominated
N total=264, independent=49

| h (s) | twin e_pos | rSim (ODE replay) e_pos | gap (m) | gap CI95 | p (non-overl.) | effect |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0077 | 0.0572 | +0.0495 | [+0.0451,+0.0524] | 1.56e-13 | +0.979 |
| 0.20 | 0.0177 | 0.0935 | +0.0758 | [+0.0665,+0.0811] | 1.24e-13 | +0.980 |
| 0.40 | 0.0382 | 0.1312 | +0.0930 | [+0.0790,+0.1029] | 4.38e-11 | +0.922 |
| 0.70 | 0.0644 | 0.1731 | +0.1087 | [+0.0877,+0.1341] | 8.18e-09 | +0.838 |
| 1.00 | 0.0919 | 0.2144 | +0.1225 | [+0.1045,+0.1452] | 1.31e-09 | +0.871 |
| 1.50 | 0.1229 | 0.2927 | +0.1699 | [+0.1188,+0.2101] | 1.03e-06 | +0.731 |

## mixed
_(sem windows)_

## rotation-dominated
N total=1, independent=0

| h (s) | twin e_pos | rSim (ODE replay) e_pos | gap (m) | gap CI95 | p (non-overl.) | effect |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0030 | 0.0881 | +0.0851 | [+0.0851,+0.0851] | 1 | +0.000 |
| 0.20 | 0.0226 | 0.1780 | +0.1554 | [+0.1554,+0.1554] | 1 | +0.000 |
| 0.40 | 0.0384 | 0.2127 | +0.1743 | [+0.1743,+0.1743] | 1 | +0.000 |
| 0.70 | 0.0250 | 0.1768 | +0.1518 | [+0.1518,+0.1518] | 1 | +0.000 |
| 1.00 | 0.0241 | 0.1791 | +0.1550 | [+0.1550,+0.1550] | 1 | +0.000 |
| 1.50 | 0.0217 | 0.1805 | +0.1588 | [+0.1588,+0.1588] | 1 | +0.000 |

## Per-bin verdicts (gates: omega placeholder + N_min population)

| bin | verdict | N indep | gap@h_ref | p@h_ref | effect@h_ref |
|---|---|---:|---:|---:|---:|
| translation-dominated | GO | 49 | +0.0930 | 4.38e-11 | +0.922 |
| mixed | INDETERMINADO (omega placeholder) | 0 | - | - | - |
| rotation-dominated | INDETERMINADO (omega placeholder) | 0 | +0.1743 | 1 | +0.000 |

## Figuras
- `rsim_twin_all_20260617_0030.png`
- `rsim_twin_trans_20260617_0030.png`
- `rsim_twin_rot_20260617_0030.png`

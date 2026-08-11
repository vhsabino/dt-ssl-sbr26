# Matched-delay ablation on rSim — 20260809_2154

The model carries an explicit transport delay and rSim does not, so part of the
gap between them could be attributed to that asymmetry alone rather than to the
identified dynamics. This ablation gives rSim the same delay, applied to its
output as a pure shift with no re-simulation, and re-scores it on identical
windows.

m = 4 samples (66.7 ms); mean translational Td = 62.8 ms.

| h (s) | model | rSim | gap vs rSim | rSim+Td | gap vs rSim+Td | delta |
|---|---|---|---|---|---|---|
| 0.10 | 0.0079 | 0.0573 | +0.0493 | 0.0932 | +0.0853 | -0.0360 |
| 0.20 | 0.0178 | 0.0924 | +0.0746 | 0.1403 | +0.1225 | -0.0478 |
| 0.40 | 0.0388 | 0.1293 | +0.0905 | 0.1822 | +0.1434 | -0.0529 |
| 0.70 | 0.0617 | 0.1744 | +0.1127 | 0.2182 | +0.1564 | -0.0437 |
| 1.00 | 0.0920 | 0.2128 | +0.1209 | 0.2495 | +0.1576 | -0.0367 |
| 1.50 | 0.1193 | 0.2891 | +0.1698 | 0.3038 | +0.1845 | -0.0147 |

N_indep = 49. Position error in metres.

Matching the delay does not close the gap; it widens it, with the same sign at
every horizon. A plant that already lags is not improved by lagging further, so
the delay term is not what separates the two. This run also reproduces the
Mode A rSim baseline of the original campaign to within 2 mm at every horizon,
which validates it against that earlier run.

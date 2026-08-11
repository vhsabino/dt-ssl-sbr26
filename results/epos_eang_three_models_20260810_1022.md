# Position and heading error, three plants, six horizons — 20260810_1022

Translation-dominated bin, medians over dense windows.

| h (s) | model e_pos | ideal e_pos | rSim e_pos | model e_ang | ideal e_ang | rSim e_ang |
|---|---|---|---|---|---|---|
| 0.10 | 0.0079 | 0.0264 | 0.0573 | 0.0128 | 0.0232 | 0.0258 |
| 0.20 | 0.0178 | 0.0501 | 0.0924 | 0.0311 | 0.0442 | 0.0464 |
| 0.40 | 0.0388 | 0.1005 | 0.1293 | 0.0440 | 0.0668 | 0.0635 |
| 0.70 | 0.0617 | 0.1516 | 0.1744 | 0.0627 | 0.0757 | 0.0792 |
| 1.00 | 0.0920 | 0.2165 | 0.2128 | 0.0733 | 0.0921 | 0.0859 |
| 1.50 | 0.1193 | 0.2910 | 0.2891 | 0.0895 | 0.0953 | 0.0936 |

Position error in metres, heading error in radians.

Two things worth reading off this table. The model has the lowest heading error
at every horizon, but its margin narrows from roughly half the baselines' at
0.1 s to within 0.005 rad of both by 1.5 s. And the two baselines cross in
position near 1 s: the zero-parameter ideal kinematics overtakes the untuned ODE
engine, which is why the rSim comparison is reported as context rather than as
the primary result.

# Pre-registered gate (window) vs event-clustered analysis — 20260810_1743

Same population, same h_ref = 0.4 s, translation-dominated bin. The paper
declares the change of analysis unit; both results are given here so the reader
can see that the verdict does not change and that the per-event version is the
conservative one.

| unit | n | median gap (m) | bootstrap 95% CI | rank-biserial | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|
| non-overlapping window (**pre-registered**) | 49 | +0.0541 | [+0.0251, +0.0830] | +0.633 | 1.15e-08 | GO |
| event (**reported**) | 10 | +0.0689 | [+0.0375, +0.0969] | +1.000 | 0.000977 | GO |

All four gate conditions — positive gap, p < 0.05, effect size ≥ 0.2, and a
bootstrap interval excluding zero — are satisfied under BOTH units. The
event-level analysis yields a larger p by construction (n = 10 against n = 49),
and that is precisely why it is the one reported.

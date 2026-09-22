# WEAR HASCA 2026 Challenge — Submission Log (Sep 22)

| File | LB | Recipe |
|------|-----|--------|
| sub_d1.csv | 0.61462 | limb-matched LGBM(4-fold subject-disjoint) + video PCA + fold-safe video-kNN feats; propagation smoothing + per-subject null-rate matching |
| sub_d2.csv | 0.61841 | d1 + self-training round 1 (pseudo weight .35) + NaN imputation |
| sub_d3.csv | 0.69065 | d2 logits + akhyar(0.670) votes x0.8 |
| sub_d4_v0.8_0.3.csv | 0.68086 | round-2 distillation + hon2 votes (regression) |
| sub_d5.csv | 0.68418 | d3 + hon2 x0.3 (isolates hon2 as harmful) |
| sub_d6.csv | 0.65535 | aka x1.0 + video-prior null rematch (overshoot) |
| sub_d7.csv | 0.69707 | d3 + udaken10 votes x0.25 — BEST of the day |

Best baseline before this session: sub_v8 = 0.71528.

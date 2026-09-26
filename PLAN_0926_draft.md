# WEAR @HASCA 2026 — plan for 2026-09-26 (5 submissions)

## Where we stand (2026-09-25 18:10 UTC)
- Our best public: **0.88791** (`subs/sub_e30_vote8.csv`), rank 5. #1 Nicolas Krusche 0.91633 (28 subs), #2 Mateo Allmer 0.90202, #3 localAI 0.90170, #4 Evelyn_Yang_02 0.88865.
- Gap to #1: +0.028. Gap to top 3: +0.014.

## Diagnosis (from 72 submissions)
1. The timeline decoder (link reconstruction + mrf4 + count calibration + within-subject video kNN) adds about +0.11 over a window-level model.
2. The leaders' edge is window-model quality on the TEST subjects: Nicolas' window-only submission was 0.772 on 09-22; UEC-dx2's public window-only ensemble scored 0.774. Our window blend is weaker on test.
3. What transfers to test: LightGBM-family models built differently (akhyar-style soft rebuild +0.011), full-data refits (+0.0035), external votes. What does not: video-heavy deep models (+0.057 on a train-subject fold, -0.008 on LB), decoder/link tweaks, pseudo-labels.
4. Every single new member on top of e19 lost 0.005-0.009; only votes across our own LB-scored files gained (+0.0004-0.0007).

## The big move: rebuild UEC-dx2's window ensemble and put it under our decoder
- UEC-dx2 code (public) is in `wear/uec/repo`: inertial-only LightGBM with rich features (beam weight 0.27), CNN8 + VideoMAE aux (0.22), XceptionTime inertial (0.20), two video MLPs (0.16, 0.14), video CNN (0.01). Their ensemble: val 0.7527 on held-out sbj 18-21, **public 0.774 window-only**.
- Overnight workflow `wear-uec-rebuild` (3 rebuild agents + 1 assembler) ports each member on CPU, checks fidelity against UEC's own recorded validation, fits on all subjects, then builds candidate CSVs C1-C5 through our exact decode (`src/make_probs.py` + `exp/transductive/refine_ns.py`).
- Raw train video for the video members is downloading in the background (sbj_7..21).

## Tomorrow's 5 submissions (decision tree)
| # | File | What it tests | Decision |
|---|---|---|---|
| 1 | C1 = e19 base + UEC ensemble at 0.5 | does UEC's window strength transfer through our decoder | > 0.8879: go to #2. <= 0.8879: go to #3 |
| 2 | C2 = e19 base + UEC ensemble at 1.0 (or C4, UEC as primary) | push the lever | keep the better of C1/C2 as anchor |
| 3 | C3 = e19 base + UEC inertial-only members at 0.5 | camera-safe part of UEC only | if C1 lost but C3 gains, the video members are the problem |
| 4 | best of the above with vote bonuses re-tuned (d7 0.8/1.2) or C4 | second-order tuning | |
| 5 | C5 = vote over the day's best + e30 + e24 | variance reduction, lock in | final |

Fallback if UEC members fail or all lose: keep e30 (0.88791) and e24 (0.88718) selected; spend at most 2 submissions on file votes.

## Honest odds
- Beat our 0.88791: ~50%.
- Reach top 3 (>= 0.9020 today, may rise): ~20-25%.
- Reach #1 (> 0.91633, may rise): ~10%. No sequence of 5 submissions can guarantee it.

## Final selection for the private LB
Select the best vote-based file plus the best single-recipe file (votes are less exposed to public-subset luck).

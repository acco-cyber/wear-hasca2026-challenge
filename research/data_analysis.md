# WEAR HASCA 2026 – hands-on test-data analysis

Author: research subagent, 2026-09-23. Scripts: `E:\Claude code\wear\research\scripts\*.py`; artifacts: `E:\Claude code\wear\research\artifacts\`.

DATA AVAILABILITY: `test_videomae_data.npy` finished downloading (1,127,485,568 bytes) during the analysis; every
video number below is on the FULL test set unless marked **[PREFIX]** (early debugging runs on the first K complete
rows, kept only where they add information). Train CSVs sbj_0..sbj_3 arrived in time (task 5 uses those four instead
of sbj_0/5/20); no train videomae .npy had arrived, so the frame-count check and the train-based chain simulation
(`05_train_chain_sim.py`, ready to run) are still pending. A background downloader also spawned six parallel train-npy
downloads into `data\tmp_par\` (not touched).

## 0. Bottom line

1. Within-subject id order is NOT temporal. Mean-pooled video cosine between consecutive ids equals the random-pair
   level (sbj 22: 0.496 vs 0.497; 23: 0.418 vs 0.423; 24: 0.442 vs 0.440; 25: 0.423 vs 0.426), and the inertial
   boundary gap for consecutive same-limb ids equals the random level (0.83 vs 0.79 g, ...). ids are one uniform
   shuffle across subjects and time (runs test: 8635 observed vs 8602 expected).
2. Windows do NOT overlap (0 exact boundary matches, 0 half-window overlaps, 0 duplicate windows/frames). The test
   tiles every second with stride 1 s; sbj 23/24/25 window counts equal the summed session durations exactly, sbj 22
   has 180 windows MORE than its listed sessions (5197 vs 2430+2410+177 = 5017).
3. Temporal chains CAN be reconstructed, partially, from the 15x768 frame features, and verified with the independent
   inertial modality: video-only linear-assignment edges have an inertially-verified exact-successor precision of
   0.44 / 0.60 / 0.45 / 0.38 (sbj 22/23/24/25), i.e. about half of all 1-s links are exactly right and the rest link
   to windows of the same scene/bout (the local ambiguity is t+-1..t+-3 within a quasi-static egocentric view).
   Adding the inertial boundary-gap likelihood for the 25 % of pairs that share a limb raises the held-out-verified
   precision further (section 3, script 07). Chains of 150-280 s are obtained, 20-40 % of windows sit in chains
   >= 100 s. Exact second-level ordering of a whole session is NOT achievable; segment-level (same bout) grouping is.
4. Structural priors from train (10 subjects, sbj_0..9): every subject performs all 18 activities (sbj_2 lacks
   one), each activity totals ~90-120 s per subject, mostly as 1 bout (~100 s) or 3 sets of ~30 s separated by
   10-40 s of null; the activity ORDER is soft-canonical (jogging first in 7/10, bench-dips last in 7/10, "(complex)"
   after its base exercise) but permuted per subject; total activity content is ~1550-2230 s per subject regardless
   of recording length, so null fraction = 1 - ~1830 s / length: predicted ~65 % for sbj 22, ~41 % for sbj 23,
   ~10-20 % for sbj 24/25 (overall ~43 %, matching the ~45 % LB optimum). Null must be budgeted per subject.

## 1. Shapes, dtypes, counts (full test set) – `01_inertial_meta.py`

- `test_meta_data.csv`: (12234, 3) columns `id, sbj_id, sensor_location`; ids are exactly 0..12233 in file order.
- `test_inertial_data.npy`: (12234, 50, 3) float64, no NaN, min -7.95 max 7.38, mean 0.19, std 0.73, values on a
  1e-8 grid (297,642 distinct values in the first 2000 windows: raw floats, not quantised). Per-axis mean/std:
  x 0.40/0.89, y 0.20/0.70, z -0.01/0.50. Median window mean |acc| = 1.027 g (p5 0.987, p95 1.968): units are g.
- `test_videomae_data.npy`: header shape (12234, 768, 15) float64 C-order, header offset 128 bytes -> 1,127,485,568
  bytes total. Axis check on 300 windows: mean cosine between adjacent slices along the 15-axis = 0.962, between
  adjacent slices along the 768-axis = 0.002, so axis 2 is TIME (15 frames) and axis 1 is the 768-d feature:
  transpose to (N, 15, 768). Frame L2 norm median 12.56 (p1 10.78, p99 14.00), no zero frames, no NaN.
- Per subject: 22: 5197, 23: 3128, 24: 1995, 25: 1914 (total 12234).
- Per limb: left_leg 3068, right_leg 3060, left_arm 3058, right_arm 3048.
- Subject x limb (left_arm, left_leg, right_arm, right_leg): 22: 1326/1311/1275/1285; 23: 773/792/768/795;
  24: 467/508/522/498; 25: 492/457/483/482. Chi-square limb-vs-subject p = 0.68, per-subject uniformity
  p = 0.35-0.87: the limb is an i.i.d. uniform draw per window (no exploitable limb pattern).

## 2. Is within-subject id order temporal? NO – `01_inertial_meta.py`, `02_video_order_chain.py`

Video (full set, cosine on L2-normalised features; consecutive = ids i, i+1 of the same subject):

| sbj | n | consecutive-id mean-pool cos (median) | random same-subject pairs | consecutive last-frame(i) vs first-frame(i+1) | random | within-window frame0 vs frame14 |
|---|---|---|---|---|---|---|
| 22 | 5197 | 0.496 | 0.497 | 0.460 | 0.461 | 0.876 |
| 23 | 3128 | 0.418 | 0.423 | 0.394 | 0.393 | 0.885 |
| 24 | 1995 | 0.442 | 0.440 | 0.409 | 0.409 | 0.878 |
| 25 | 1914 | 0.423 | 0.426 | 0.384 | 0.388 | 0.833 |

Consecutive ids are indistinguishable from random pairs (all differences <= 0.005), while genuinely adjacent frames
inside one window score 0.83-0.89 fourteen frames apart (and 0.966 one frame apart).

Inertial (full set; consecutive ids of the same subject AND same limb; distance = ||last sample of i - first sample of i+1||):

| sbj | #pairs | consecutive median | random same-limb median | intra-window step median | frac consecutive < step median | frac random < step median |
|---|---|---|---|---|---|---|
| 22 | 1359 | 0.828 | 0.794 | 0.030 | 0.005 | 0.005 |
| 23 | 813 | 1.293 | 1.290 | 0.049 | 0.004 | 0.007 |
| 24 | 496 | 1.308 | 1.279 | 0.050 | 0.006 | 0.004 |
| 25 | 484 | 1.242 | 1.271 | 0.062 | 0.004 | 0.004 |

Both modalities agree: the id order carries no temporal information. Artifacts:
`artifacts/inertial_consec_continuity.csv`, `artifacts/video_consec_similarity.csv`.

## 3. Chain reconstruction feasibility – `02_video_order_chain.py`, `06_chain_v2.py`

Setup: each frame L2-normalised; descriptors tail = mean of last 3 frames, head = mean of first 3 frames (also
last1/first1, symmetric mean-pool and symmetric concat-15). Directed score S[A,B] = cos(tail_A, head_B). Greedy
chaining (edges by descending score, <=1 successor and <=1 predecessor, union-find to forbid cycles) or linear
assignment (Hungarian on -LLR, then cut the weakest edge of every cycle and the bottom 3 % of edge scores).
The only ground-truth-free but INDEPENDENT check available is inertial: for an edge A->B whose windows carry the
same limb, a true adjacency means B's first sample is one 20 ms step after A's last sample, so the gap should look
like an intra-window step (median 0.03-0.06 g), not like a random same-limb pair (median 0.8-1.3 g). The edge
precision estimate is (frac_edge_gap<2*step - frac_random<2*step) / (frac_trueadj<2*step - frac_random<2*step),
with the true-adjacent rate taken from intra-window steps (0.64-0.67).

### 3a. Nearest-neighbour statistics (full set, `02_video_order_chain.py`, `artifacts/video_chain_stats.csv`)

| sbj | descriptor | best-match cos median (p10/p90) | best-2nd margin median | mutual best | greedy chains | longest | nodes in chains>=100 | same-limb edges: gap median vs random | frac<2 step vs random |
|---|---|---|---|---|---|---|---|---|---|
| 22 | last3/first3 (directed) | 0.924 (0.873/0.957) | 0.0105 | 0.441 | 176 | 279 | 32 % | 0.178 vs 0.795 | 0.309 vs 0.020 |
| 23 | last3/first3 | 0.932 (0.865/0.971) | 0.0098 | 0.442 | 101 | 246 | 33 % | 0.239 vs 1.277 | 0.394 vs 0.017 |
| 24 | last3/first3 | 0.933 (0.859/0.971) | 0.0101 | 0.434 | 58 | 177 | 18 % | 0.264 vs 1.270 | 0.301 vs 0.013 |
| 25 | last3/first3 | 0.896 (0.837/0.950) | 0.0144 | 0.448 | 34 | 239 | 65 % | 0.473 vs 1.272 | 0.269 vs 0.011 |
| 22 | sym mean-pool | 0.940 (0.895/0.969) | 0.0086 | 0.461 | 73 paths | 718 | 80 % | 0.150 | 0.319 |
| 23 | sym mean-pool | 0.950 | 0.0085 | 0.478 | 28 | 362 | 86 % | 0.172 | 0.418 |
| 24 | sym mean-pool | 0.952 | 0.0083 | 0.488 | 17 | 444 | 98 % | 0.281 | 0.300 |
| 25 | sym mean-pool | 0.915 | 0.0118 | 0.456 | 11 | 476 | 99 % | 0.406 | 0.248 |
| 22 | sym concat-15 | 0.886 (0.827/0.932) | 0.0095 | 0.416 | 120 | 564 | 75 % | 0.160 | 0.307 |
| 23 | sym concat-15 | 0.901 | 0.0092 | 0.445 | 41 | 497 | 87 % | 0.170 | 0.418 |
| 24 | sym concat-15 | 0.905 | 0.0096 | 0.433 | 16 | 684 | 93 % | 0.277 | 0.298 |
| 25 | sym concat-15 | 0.855 | 0.0131 | 0.416 | 11 | 554 | 98 % | 0.414 | 0.246 |

(last1/first1 is uniformly slightly worse than last3/first3: mutual 0.42, longest 271/255/221/335.)
Best-vs-second margins of ~0.01 and mutual-best rates of 42-49 % mean LOCAL ordering is ambiguous: the egocentric
scene is quasi-static during stretching / push-ups / sit-ups / bench-dips so t+-1, t+-2, ... all score alike.
Chain-length histogram, directed greedy last3/first3 (bins of 10 s): sbj 22: 68 chains <10 s, 29 in 10-19, 22 in
20-29, 14, 9, 12, 4, 1, 4, 2 up to 99, then 1/5/2 in 100-129, 1 each at 190, 230, 270 (longest 194/239/279);
sbj 23: 38/16/7/11/12/1/3/1/2/2 up to 99, 4 in 100-109, 2 in 110-119, 1 at 130, 1 at 240 (longest 136/246);
sbj 24: 25/6/5/2/2/3/5/2/3/3 up to 99, 2 at 170 (longest 176/177); sbj 25: 9/2/6/2/4/1/1 up to 69, then 1 each at
90/100/130, 2 at 140, 2 at 150, 1 at 170, 1 at 230 (longest 173/239).

### 3b. Linear-assignment chains and INDEPENDENT verification (`06_chain_v2.py`, `07_chain_heldout.py`)

Video-only (LLR from tail3/head3 cosine; verification = full 3-axis inertial boundary gap on same-limb edges):

| sbj | n | edges | cycles cut | chains | longest | nodes in chains>=100 | same-limb edges | gap median | frac<2 step | random | true-adj proxy | est. exact-successor precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 5197 | 4753 | 303 | 444 | 195 | 37 % | 1222 | 0.190 | 0.304 | 0.018 | 0.669 | 0.44 |
| 23 | 3128 | 2789 | 265 | 339 | 193 | 18 % | 691 | 0.199 | 0.398 | 0.016 | 0.648 | 0.60 |
| 24 | 1995 | 1813 | 128 | 182 | 181 | 34 % | 486 | 0.295 | 0.298 | 0.013 | 0.645 | 0.45 |
| 25 | 1914 | 1759 | 98 | 155 | 171 | 9 % | 475 | 0.447 | 0.259 | 0.012 | 0.661 | 0.38 |

Video + inertial LLR with a HELD-OUT verifier (inertial term scored on the x,y boundary gap only, verified with the
z-axis gap |last_z(A) - first_z(B)| < 2 x median intra-window z step; random same-limb pairs pass that test 5.5-8.4 %
of the time, true consecutive samples 63-67 %):

| sbj | scoring | edges | chains | longest | nodes in chains>=100 | same-limb edges | z-gap median | frac pass | held-out precision | precision of top-20 % scored edges | bottom-50 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | video only | 4739 | 458 | 251 | 31 % | 1214 | 0.061 | 0.324 | 0.45 | 0.73 | 0.25 |
| 22 | video + xy-inertial | 4708 | 489 | 226 | 30 % | 2004 | 0.028 | 0.445 | 0.65 | 0.91 | 0.30 |
| 23 | video only | 2762 | 366 | 196 | 12 % | 700 | 0.072 | 0.400 | 0.60 | 0.91 | 0.36 |
| 23 | video + xy-inertial | 2679 | 449 | 128 | 8 % | 1511 | 0.034 | 0.510 | 0.79 | ~1.0 (1.19, sampling noise) | 0.30 |
| 24 | video only | 1804 | 191 | 139 | 30 % | 467 | 0.108 | 0.330 | 0.46 | 0.69 | 0.35 |
| 24 | video + xy-inertial | 1768 | 227 | 276 | 20 % | 966 | 0.046 | 0.452 | 0.67 | 0.94 | 0.32 |
| 25 | video only | 1775 | 139 | 228 | 29 % | 451 | 0.130 | 0.328 | 0.41 | 0.47 | 0.43 |
| 25 | video + xy-inertial | 1758 | 156 | 222 | 20 % | 836 | 0.060 | 0.449 | 0.62 | 0.81 | 0.26 |

(The circular variant in `06_chain_v2.py`, scoring AND verifying with the full 3-axis gap, gives frac<2 step
0.69/0.77/0.66/0.76, i.e. at/above the true-adjacent proxy, which is why the held-out design above is the one to
quote.) Full-set LSA chain-length histogram (video+inertial, bins of 50): sbj 22: 455 chains <50, 16 in 50-99, 11 in
100-149, 1 in 150-199, 2 >= 200; sbj 23: 431/12/3; sbj 24: 211/4/3/1; sbj 25: 171/6/1/2.

Interpretation / how a temporal ordering can be reconstructed:
- Exact second-by-second ordering of whole sessions is NOT recoverable: even with both modalities ~1/3 of the links
  are off by a few seconds and chains break every ~10-20 s on average (median chain length 2-3, mean ~10).
- But the links are far from random: 45-60 % (video only) and 62-79 % (video + inertial) of links are EXACT
  successors, the top-20 % scored links are 81-94 % exact, and wrong links stay inside the same scene/bout (mean-pool
  NN paths cover 75-99 % of windows in paths >= 100 nodes). So chains are reliable as ORDERED SEGMENTS: propagate
  per-window class probabilities along them (HMM / median filter with a confidence-weighted transition = the LLR
  score), and cluster chains into bouts.
- Ordering is only useful relative to the label structure: in train, each activity is 1-3 contiguous sets of
  ~30-100 s (section 5) inside a null-free or short-rest region, so a chain of 100+ s is either one bout or
  bout+rest+bout of the SAME activity. The exploitable constraint is therefore "each activity class occupies one
  compact region of the timeline per subject" (assignment of 18 classes to segments with total ~90-120 s each), not
  "one contiguous bout per class".
- Recommended: (i) score = LLR_video(tail3/head3) + LLR_inertial(same-limb boundary gap, 3 axes now that no held-out
  is needed) [+ linear-extrapolation LLR of the last 5 inertial samples]; (ii) linear assignment per subject;
  (iii) cut cycles and the lowest-scored links, keep the link score as an edge confidence; (iv) chain-level smoothing
  of class probabilities; (v) per-subject class-to-segment assignment with per-subject null budget (section 5/6).
- Definitive validation script (needs one train videomae npy): `05_train_chain_sim.py sbj_0` simulates the test
  construction on a train subject with known order and reports EXACT successor precision and label purity.

Artifacts: `artifacts/chain_successor_video_xyinertial.csv` (id, sbj_id, succ_id; -1 = chain end; full set),
`artifacts/chain_heldout_summary.csv`, `artifacts/chainv2_summary.csv`,
`artifacts/chainv2_succ_sbj{22,23,24,25}_{video_only,video_inertial}.npy` (successor test-id per window),
`artifacts/chain_succ_sbj*_last3_first3.npy`, `artifacts/chain_bestsim_sbj*_last3_first3.npy`,
`artifacts/video_chain_top_chains.csv` (first 30 ids of the 5 longest greedy chains per subject),
`artifacts/video_chain_stats.csv`, `artifacts/video_meanpool.npy` (12234 x 768 float32), plus the `*_partial*`
files from the prefix debugging runs.

## 4. Inertial boundary sharing – NONE – `01_inertial_meta.py`

Same subject + same limb, first sample of j == last sample of i (rounded to 1e-6): 0 matches in all 16
subject x limb groups (n = 457..1326 each). Half-window overlap (last 25 samples of i == first 25 of j, 1e-5): 0.
Nearest-neighbour distance last[i] -> first[j] has median 0.024-0.092 and p10 0.003-0.024, only ~0-0.9 % below 1e-3,
consistent with dense 3-d point clouds, not with shared samples. Exact duplicate windows: 0; constant windows: 0.
Conclusion: windows are disjoint 1-s tiles (samples 50t..50t+49); adjacent windows are 20 ms apart, which is
exactly why the boundary-gap LLR in section 3 works as a verifier but never as an exact-equality key.
Artifacts: `artifacts/inertial_boundary_match.csv`.

## 5. Train structure – `03_train_structure.py 0 1 2 3` (`artifacts/train_label_runs.csv`)

CSV format: 14 columns (`sbj_id`, 12 acc columns, `label`), 50 Hz, one file = one subject's full recording; null is
stored as NaN in `label` (sbj_0: 28,189 NaN rows = 20.2 %), no NaN in the acc columns.

| sbj | samples | duration | null frac | non-null bouts | classes | bouts/class (mode) | typical bout | total per class | null runs: n, median, max |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 139,725 | 2794.5 s (46.6 min) | 0.202 | 33 | 18 | 1-3 | 30-100 s (1-bout classes 76-183 s) | 76-218 s | 34, 13.5 s, 52 s |
| 1 | 138,900 | 2778.0 s (46.3 min) | 0.352 | 37 | 18 | 3 (1 for 8 classes) | 23-58 s sets / 84-119 s single | 62-144 s | 37, 23.4 s, 65 s |
| 2 | 178,200 | 3564.0 s (59.4 min) | 0.445 | 48 | 17 (no "jogging (rotating arms)") | 3 (1 for 3 classes, 4 for 4) | 17-80 s | 84-225 s | 49, 27.2 s, 119 s |
| 3 | 208,550 | 4171.0 s (69.5 min) | 0.551 | 59 | 18 | 3 (16 of 18 classes; sit-ups 6, skipping 7) | 28-32 s | 83-182 s (mostly 85-100) | 60, 41.5 s, 60 s |

- The protocol is 3 SETS of ~30 s per activity with 10-40 s rest, sometimes collapsed into one ~90-110 s bout
  (all stretching variants in sbj 0/1, lunges in sbj 0/1, 8 classes in sbj 1). So an activity is NOT "one contiguous
  bout": it is a compact region of 1-4 bouts (rarely up to 7 with label glitches: sbj_3 has 3 non-null bouts < 3 s
  such as a 1.38 s bench-dips and a 2.5 s skipping fragment) whose non-null total is ~90-120 s per subject.
- Each activity appears once per subject as such a region (sbj_2 lacks one class entirely, so 17 regions). The
  ORDER is different for every subject (all 6 pairwise comparisons of the order differ; e.g. sbj_0 starts jogging,
  triceps stretch, rotating-arms, lunging stretch...; sbj_1 starts sidesteps, shoulders stretch, butt-kicks...;
  sbj_3 starts sidesteps, bench-dips, shoulders stretch...), so no order prior can be transferred. Consecutive regions
  are often related (jogging variants back-to-back in sbj_2, push-ups then push-ups (complex), sit-ups then sit-ups
  (complex)), but not consistently.
- Bout duration per class over the 4 subjects (median / max, s): bench-dips 29/40, burpees 29/49, jogging 58/98,
  butt-kicks 30/66, rotating arms 94/97, sidesteps 51/105, skipping 31/57, lunges 30/173, lunges (complex) 32/183,
  push-ups 30/37, push-ups (complex) 32/59, sit-ups 31/96, sit-ups (complex) 43/100, stretching 41-56 / 92-119.
- Null fraction grows with recording length (0.20 at 46 min ... 0.55 at 70 min) because the activity content is
  roughly constant (~1800-2230 s per subject) and the rest of the recording is null (walking between spots, rests).
  Implication for the test subjects (18 activities each): sbj 22 (5197 s) should be ~55-65 % null, sbj 23 (3128 s)
  ~30-40 %, sbj 24 (1995 s) and sbj 25 (1914 s) cannot even hold 1800 s of activity at train pacing, so their sets
  were shorter and null is probably 15-35 %. The overall optimum of ~45 % found on the LB last session is consistent
  with this mix (e.g. 60 % / 35 % / 30 % / 30 % gives 45.7 %), and the per-subject values must differ by a factor of two.
- Six more subjects (`03_train_structure.py 5 4 6 7 8 9`, summary lines only): duration / null fraction / non-null
  bouts: sbj_5 35.8 min / 0.278 / 36; sbj_4 53.2 / 0.465 / 37; sbj_6 41.7 / 0.267 / 27; sbj_7 47.6 / 0.397 / 29;
  sbj_8 41.6 / 0.286 / 23; sbj_9 36.1 / 0.192 / 24; all 18 classes present in each. Activity content
  (duration x (1 - null)) over the 10 subjects is 1553-2231 s (mean ~1830 s), so null fraction is essentially
  1 - 1830 s / recording length: predicted test null ~65 % (sbj 22, 5197 s), ~41 % (sbj 23, 3128 s) and only
  ~10-20 % for sbj 24/25 (1995 s / 1914 s), overall ~43 %, matching the ~45 % LB optimum found last session.
- Order prior, revised with 10 subjects: it is NOT fixed (all 45 pairwise orders differ) but it is soft-canonical:
  7/10 start with plain jogging, then usually jogging (rotating arms) -> jogging (skipping) -> stretching (triceps) ->
  push-ups -> sit-ups -> ... -> burpees -> stretching (hamstrings/lumbar) -> lunges (complex) -> bench-dips, which is
  LAST in 7/10; the "(complex)" variant usually follows its base exercise. sbj_1 and sbj_3 start with sidesteps and
  deviate most. Some subjects interleave sets of different exercises (sbj_8: push-ups, sit-ups, push-ups, sit-ups
  (complex), ..., sit-ups, push-ups), so "one compact region per class" holds for most but not all classes.
- Null runs between bouts: median 13-42 s per subject (min 0.7 s, max 196 s), first/last null of a recording
  0.7-5 s / 3-119 s: recordings start and end almost immediately with activity.
- Video frame count vs inertial length: no train videomae .npy was on disk yet (only 0-byte placeholders in
  `data\tmp_par\`); the public baseline config (`kernels/akhyar2612__0-670`) documents 30 fps frames aligned to 50 Hz
  samples (0.6 frames per sample, window frames = video_start+8 .. +22), and `03_train_structure.py` prints the ratio
  as soon as a file exists.
- Cross-check with public-kernel outputs (`kernels/abhinavm2811__.../*.ipynb`, stride-25 windows):
- 277,304 half-second windows over the 24 files; null = 110,140 (39.7 %); each of the 18 activity classes has
  8,448-10,012 windows (3.0-3.6 % each), i.e. ~176-209 s of every activity per file, so ~3-3.5 min bouts in train.
- Windows per file: sbj_0 11176, sbj_0_2 8964, sbj_1 11112, sbj_10 15940, sbj_11 8888, sbj_12 13080, sbj_13 15912,
  sbj_14 12804, sbj_14_2 13420, sbj_15 10412, sbj_16 14040, sbj_17 12364, sbj_18 10528, sbj_19 8436, sbj_2 14256,
  sbj_20 7632, sbj_21 10224, sbj_3 16684, sbj_4 12764, sbj_5 8600, sbj_6 10012, sbj_7 11412, sbj_8 9980, sbj_9 8664
  (x0.5 s = 64-139 min per file).
  (the per-file 176-209 s per class is over stride-0.5 s windows, i.e. ~90-105 s of activity per class per file,
  matching the bout arithmetic above).

## 6. Other exploitable structure – `04_extras.py`

- sbj 22 has 180 windows more than its listed sessions (5197 vs 5017 s). 180 = 3:00 exactly; session_3 is 2:57.
  Either a duplicated/unlisted 3-min segment or rounded session lengths; a 4th chain of ~180 s in the chain
  reconstruction would confirm it.
- Limb choice is i.i.d. uniform per window (chi-square p 0.35-0.87); limb sequence runs 9197 vs 9176 expected.
- Sensor mounting is consistent across test subjects: gravity direction per limb (mean acc, g) is
  left_arm (-0.49..-0.64, +0.24..+0.43, -0.09..-0.35), right_arm (+0.45..+0.63, +0.32..+0.50, -0.11..-0.33),
  legs (+0.71..+0.86, ~0, +0.12..+0.24) for all four subjects, so limb-specific models transfer; left/right arms are
  mirrored in x. Window mean |acc| per subject x limb 1.07-1.30 g.
- Motion level differs by subject: fraction of near-static windows (std < 0.05 g) 0.36 / 0.38 / 0.30 / 0.21 for
  sbj 22/23/24/25, matching the shorter, denser sessions of sbj 24/25 (less null).
- No edge artefacts: mean |diff| by sample position is flat (0.081-0.083) so windows are cut from one continuous
  stream without filtering restarts.
- No duplicated video windows (0) and no duplicated boundary frames (0) in the full set: each second's video appears
  once, so no cross-limb video duplication to exploit.
- The successor links themselves are an exploitable feature: a window's neighbours in the reconstructed chain (and
  their limbs) give a multi-limb, multi-second context that the official single-limb 1-s window lacks; with ~2/3 of
  links exact and the rest within the same bout, chain-level pooling of IMU features per limb is a cheap way to
  recover most of the 4-limb train signal at test time.
- Video frames inside a window are strongly autocorrelated (adjacent cos 0.966 median, frame0-frame14 0.87), so
  mean/std pooling loses little for classification but ALL of the ordering information; the previous pipeline
  discarded exactly the part that enables chains.

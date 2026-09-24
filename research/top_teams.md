# 3rd WEAR Dataset Challenge @HASCA 2026 -- competitive intelligence (top teams)

Date: 2026-09-23. Source LB snapshot: `C:\Users\Koushik\AppData\Local\Temp\claude\E--Claude-code\4892a651-709e-47dc-8a3d-bbb791e079fe\scratchpad\lb.csv` (131 ranked teams + 4 organiser baselines, 2,693 submissions total per the Kaggle overview page).

## 1. Leaderboard structure

### 1.1 Distribution (131 ranked teams, public macro-F1)
- mean 0.582, median 0.632, p25 0.552, p75 0.720, max 0.88425.
- Organiser baselines (rank 0, inertial only): AttendAndDiscriminate 0.56091, TinyHAR 0.54594, DeepConvLSTM 0.53136, random 0.05263. Anything above ~0.56 is therefore "used the video features" territory; the video modality alone is what lifts scores past 0.60.

### 1.2 Clusters (score band: n teams, median subs, mean subs)
| band | n | median subs | mean subs | who |
|---|---|---|---|---|
| 0.84-0.885 | 10 | 24.5 | 36.0 | thisray 0.88425 (6 subs), An Dao 0.88126 (4), Beaujard 0.87430 (29), habakan 0.87059 (14), anhnam_xtanh 0.86687 (41), T Hansda 0.86584 (20), Vedanam Coders 0.86132 (97), Gleb Shanshin 0.85796 (76), KMET 0.84551 (69), Mateo Allmer 0.84061 (4) |
| 0.77-0.84 | 15 | 18 | 47.3 | Rajveer 0.816 (6), ChinMoroc 0.813 (103), Fengwx 0.811 (49), Yeashu 0.809 (16), [Deleted] 0.808, Dale Zhong 0.791, Chankin 0.789, bigsuperplum 0.788 (2), Randy 0.785, BABW 0.780, Use Responsible AI 0.779, UEC-dx2 0.774 (9 members, 57 subs), Third_time_is_a_charm 0.773 (192 subs), Nicolas Krusche 0.772 (1 sub), **Whatever 0.77190 (188 subs)** |
| 0.72-0.77 | 8 | 12 | 30 | sdssJ0, Le Trong Hieu, berrylyte, Rister, Dong, HARMA 0.737, HCMUT 0.722 (134 subs) |
| 0.66-0.72 | 23 | 8 | 20.8 | includes **us (Koushik Rudra 0.71528, 42 subs, rank 37)**, and a 5-team plateau at exactly 0.66456 |
| 0.60-0.66 | 32 | 3.5 | 9.3 | mostly copies of public kernels |
| < 0.60 | 43 | | | baseline-level or broken submissions |

Observations:
- There is a clear gap between 0.84061 (rank 10) and 0.81601 (rank 11), and another between 0.77190 (rank 25) and 0.74431 (rank 26). Three tiers: **0.84-0.885 "solved the structure"**, **0.77-0.82 "strong multimodal model"**, **0.72-0.74 "good tabular/video model"**.
- The 5 teams tied at exactly 0.66456 (akhyar2612, rupsarroy, honghanhhh, mdaburummanrefat, sibamsamanta07) are verbatim reruns of the public kernel `akhyar2612/0-670` (LightGBM + PCA-pooled video + sensor_location + class_weight). That is the public-kernel ceiling: ~0.665-0.67.
- The most-voted public kernel (`nomannic19/ts-emb-3wdc-temporal-fusion-ensemble`, 11 votes: Inception 1D-CNN on IMU + 1-layer Transformer over the 15 VideoMAE frames + pooled-fusion model, label smoothing, modality dropout) scored only **0.61278** for its author (rank 77). End-to-end small neural fusion on random 1-s windows is NOT what the top is doing.

### 1.3 Submission count vs score
- Spearman(score, submissions) = **0.51** over the 131 teams, but this is driven by the bottom (1-2 subs = broken/baseline). Inside the top 10 the relation inverts: the four highest scores used 6, 4, 29 and 14 submissions; the three July grinders (97, 76, 69 subs) sit at ranks 7-9. Hard-grinding does not buy the last 3 points; a specific insight does.
- 4 of the top 10 have <= 14 submissions and all 4 arrived in September (thisray 09-22, An Dao 09-19, habakan 09-22, Allmer 09-22, whose account was created 12 days ago). Several appear to have found the recipe in days: **the 0.84-0.88 recipe is reproducible from public information plus the test-set structure**, it is not a months-long tuning exercise.

### 1.4 Dates: July grinders vs September few-shot entrants
- The Kaggle overview page states: "HASCA-WEAR paper submission: July 5 9:00 AoE, 2026" and "Prizes will be awarded to the three teams with the highest **private** leaderboard score at the time of the paper submission deadline." Marius Bock posted "Technical Report Deadline Passed" ~3 months ago (URL below). So the prize race ended on July 5; the Kaggle LB stays open until 2026-10-12 only as an open leaderboard.
- July cohort (last submission <= 2026-07-19): Vedanam Coders 0.86132 (07-07), Gleb Shanshin 0.85796 (07-05), KMET 0.84551 (07-04), ChinMoroc 0.81282, Fengwx 0.81060, [Deleted] 0.80780 (07-03), UEC-dx2 0.77373, Third_time_is_a_charm 0.77288, Whatever 0.77190 (07-05), HARMA 0.73709, HCMUT 0.72210. **The workshop prize winners (on private LB) are almost certainly among Vedanam Coders / Gleb Shanshin / KMET**, and their technical reports will appear in the UbiComp/ISWC 2026 Adjunct Proceedings (HASCA) around October 11-12, i.e. AFTER the Kaggle close. No 2026 report is on arXiv yet (searched arXiv for "WEAR" + VideoMAE + HASCA 2026: nothing).
- Month medians of last-submission date: Jul 0.773 (n=16), Aug 0.611 (n=26), Sep 0.640 (n=65) but Sep contains the 4 new top-5 entries. September entrants are bimodal: either public-kernel copies (~0.61-0.66) or 0.84-0.88 with a handful of submissions.

## 2. Who the top-10 are and what can be inferred

Kaggle API (`kaggle kernels list --user`, `kaggle datasets list --user`) and profile pages fetched 2026-09-23. **None of the top 10 has published any code, dataset, or discussion for this competition.** The competition discussion forum has only 3 threads (host announcements + one data question), no solution write-ups.

| rank | team / user | profile facts | inferred |
|---|---|---|---|
| 1 | thisray (0.88425, 6 subs, 09-22) | https://www.kaggle.com/thisray -- ML engineer at MediaTek, Hsinchu; 42 competitions, 4 writeups; recent public work: "S6E8 Addiction Blend LB 0.97117" (Playground blend notebook, Aug 2026), "NeuroGolf 4808.21" post-processing notebook (Apr 2026) with a "submission zip and task table" dataset. Follows CPMP, Chris Deotte. | A blend/post-processing specialist. Reaching 0.884 in 6 submissions suggests he did not train for weeks -- consistent with a structural/post-processing trick on top of a decent base model (see section 4). |
| 2 | An Dao / andao94 (0.88126, 4 subs, 09-19) | https://www.kaggle.com/andao94 -- Hanoi; 3 competitions; datasets are "KeyFrame_part*" (video keyframes, 2024, ~26 GB) -> video-retrieval background (likely the Vietnamese AI Challenge / HCMC event-retrieval). | Video-embedding practitioner. 4 submissions to 0.881 again implies the gain is not hyper-parameter grinding. |
| 3 | Benjamin Beaujard (0.87430, 29 subs) | https://www.kaggle.com/benjaminbeaujard -- account created a month ago, 3 competitions, nothing public. | Unknown. |
| 4 | habakan / kansukehabano (0.87059, 14 subs) | https://www.kaggle.com/kansukehabano -- Competitions Expert (rank 2,681), Tokyo, CEO/CTO METRICA; GitHub https://github.com/habakan (WASM/probabilistic-programming repos, nothing HAR). Also active in Biohub cell tracking ("biohub exp007 per-movie caps" kernel, Aug 2026). | Experienced Kaggler; 14 subs to 0.871. |
| 5 | anhnam_xtanh (0.86687, 41 subs, 09-10) | https://www.kaggle.com/anhnamxtanh -- datasets are Phoenix-2014 / Phoenix-2014-T (sign-language video), i.e. video-sequence modelling background. | Video sequence modeller. |
| 6 | T Hansda (0.86584, 20 subs, 09-22) | https://www.kaggle.com/hansda -- IIT Kharagpur student, 7 competitions, nothing public relevant. | |
| 7 | Vedanam Coders (0.86132, 97 subs, 07-07) | huynhtruongtu (Kyushu Institute of Technology student) + umangdobhal03. Same team name published last year: "Two-Stage Reservoir Computing for Sensor-Specific Activity Recognition Using the WEAR Inertial Dataset" (UbiComp Companion 2025, https://dl.acm.org/doi/10.1145/3714394.3756190): CNN + reservoir computing, sensor-specific models; last-year public LB 0.53017 (rank 15). | Prize-race entrant; wrote a technical report by July 5. Their 2026 report will be in the HASCA 2026 proceedings. |
| 8 | Gleb Shanshin / gleb270 (0.85796, 76 subs, 07-05) | https://www.kaggle.com/gleb270 -- Competitions Master, quant researcher (Pinely), ex-VK; public "4th place DRW Crypto Prediction" notebook. | Strong general Kaggler, tabular/GBDT + time-series. Prize-race entrant. |
| 9 | KMET (0.84551, 69 subs, 07-04) | julialasek + mateuszdanio (Polish names; nothing public). | Prize-race entrant. |
| 10 | Mateo Allmer (0.84061, 4 subs, 09-22) | account created 12 days ago, 6 competitions joined. | Reached 0.84 in 4 submissions -> likely same structural recipe. |

Notable non-top-10:
- **Whatever (rank 25, 0.77190, 188 subs)** = francalatrava, leishistuttgart, vitorfortesrey = DFKI/Lukowicz group. They **won last year** (2nd challenge, public 0.60249) with "FAME: Feature-Augmented Multi-View Ensemble Framework for HAR using Inertial Sensors" (https://doi.org/10.1145/3714394.3756194, the report the organisers link as "winners of last year's challenge"). Their best hand-crafted-feature multi-view GBDT ensemble + video only reaches 0.772 this year after 188 submissions. **This is the strongest evidence that the 0.84-0.88 group is not doing "better features/models" but exploiting the test-set structure.**
- HARMA (ricardalink, 0.73709) published "Mitigating Null-Class Dominance..." last year (https://dl.acm.org/doi/10.1145/3714394.3756192).
- Nomannic (0.61278) is the author of the most-voted public kernel (temporal fusion ensemble) -- proof that a straightforward deep fusion model on isolated 1-s windows lands at ~0.61.

## 3. What was achievable last year (2nd WEAR challenge, inertial only)
Public LB via `kaggle competitions leaderboard 2nd-wear-dataset-challenge -s` (discussion forum is empty, 1 public kernel):
1. Whatever 0.60249, 2. Schw1Eg0 of the White Forest 0.58750, 3. HARMA 0.57773, 4. Signal Sleuths 0.57657, 5. Sumitou Atsuya 0.57485, 6. 3KA 0.57441, 7. Trina Sarkar53 0.56028, 8. Michael Ibrahim 0.55915, ... 15. VedaNam Coders 0.53017.
Published 2025 reports: FAME (Whatever, winner), HARMA null-class mitigation, VedaNam two-stage reservoir computing, 3KA "Robust In-the-Wild Exercise Recognition from a Single Wearable: Data-Side Fusion, Sensor Rotation, and Feature Engineering" (arXiv 2511.23173: 450 hand-crafted features, left/right limb data fusion, axis-inversion + 180-degree rotation augmentation, HistGB(balanced)+XGB soft vote, group-5-fold macro-F1 58.83%), Chandankar & Burchard "Sensor-Specific PatchTST Ensembles with Test-Matched Augmentation" (arXiv 2510.21282), "Challenging High-Performance HAR with a SOTA model and simple preprocessing" (10.1145/3714394.3756191). 2024 (4-sensor version): Signal Sleuths left-right swapping + upper-lower pairing, 3-fold CV 91.87% (arXiv 2408.03947).
Take-away: inertial-only single-sensor 1-s windows saturate at ~0.60 macro-F1 even for the DFKI group. This year the same group + video reaches 0.77. Everything from 0.77 to 0.88 must come from the video modality and/or the test-set structure.

## 4. Inference: what the 0.84-0.88 recipe most likely is
Verified facts that make a structural exploit available (from the task context, not re-derived here): the test tiles every second of every session with stride 1 s (window counts equal summed session durations exactly); each test subject performs each of the 18 activities exactly once as a contiguous bout; the 15 VideoMAE frame features are the central frames of a 30-frame second; ids are shuffled.

1. **Re-order the test windows in time using the video features.** Consecutive 1-s windows share overlapping VideoMAE clips (frame i uses clip i-8..i+7, so adjacent seconds' central-15 frames are 16-frame-clip neighbours): the 15x768 sequence of window t and window t+1 are near-continuations. Nearest-neighbour chaining on the last/first frame features (or on the full 15x768 sequence, e.g. cosine similarity between frames 8-14 of window t and frames 0-6 of window t+1) can reconstruct each subject's timeline. Once ordered, the problem becomes segmentation of a ~30-50-minute video with exactly 18 activity bouts + null gaps -- which is trivially smoothed (HMM/Viterbi with the "each activity once" constraint, median filtering, change-point detection). This alone explains (a) the 0.84-0.88 jump, (b) why entrants with 4-6 submissions get there, (c) why the DFKI inertial-feature experts are stuck at 0.77 with 188 submissions.
2. **Use the temporal structure inside the 15 frames** (our pipeline only mean/std-pooled them). The top public kernel already attends over the 15 frames; the 0.77-0.82 tier probably comes from sequence models over the 15x768 plus IMU.
3. **Bout-level decoding**: per subject, 18 bouts each appearing once; assign labels by Hungarian/Viterbi over segments with per-class priors (activity durations are similar across subjects in train). Macro-F1 rewards fixing the confusable jogging/stretching variants (1-5, 6-10) at the bout level rather than per window.
4. **Null-rate matching** (already in our pipeline) remains necessary; ~40% of the timeline is null in train.

Confidence: the exploitability of stride-1 tiling is a verified fact from the task context; that the leaders use it is an inference from (i) submission counts, (ii) Whatever's plateau, (iii) absence of any modelling insight in public code. No leader has confirmed publicly.

## 5. Public code / datasets / write-ups found
- Kaggle public kernels (3rd challenge): `nomannic19/ts-emb-3wdc-temporal-fusion-ensemble` (11 votes, author LB 0.61278), `akhyar2612/0-670` (LightGBM + PCA video pooling, LB 0.66456 reproduced by 5 accounts), `honghanhhh/wear-hasca` (LightGBM, 0.66456), `binasalama/wear-hasca-baseline`, `udaken10/base-line-liner-model`, `stmugiwara/*` (CNN-LSTM torch, scores 0.04 = broken). Pulled copies: `C:\Users\Koushik\AppData\Local\Temp\claude\E--Claude-code\4892a651-709e-47dc-8a3d-bbb791e079fe\scratchpad\kernels\`.
- GitHub: https://github.com/Agnuxo1/wear-hasca-2026 (starter: schema inspection, participant-aware split, no scores), https://github.com/acco-cyber/wear-hasca2026-challenge (our own repo, LightGBM subject-disjoint CV, null calibration).
- Organiser pages: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026 (overview, timeline, prizes), discussion threads /discussion/720908 ("Technical Report Deadline Passed"), /discussion/738942 (train VideoMAE .npy missing in latest dataset version -- Bina Salama, 20 d ago), https://mariusbock.github.io/wear/challenge.html, LinkedIn announcement by Kristof Van Laerhoven.
- Last year: leaderboard via Kaggle API (above); FAME winner report https://doi.org/10.1145/3714394.3756194; HASCA 2025 TOC https://dl.acm.org/doi/proceedings/10.1145/3714394?tocHeading=heading16.
- Nothing from thisray / An Dao / Beaujard / habakan / Hansda / Allmer on this competition anywhere public (Kaggle code/datasets/discussions, GitHub, web search).

## 6. Open questions
- Is within-subject id order temporal? (If yes, ordering is free; if not, video-NN chaining is required.) Not checked here -- must be tested on `test_videomae_data.npy`.
- Which of Vedanam/Gleb/KMET actually won the private-LB prize race; their reports surface only at HASCA on Oct 11-12, after the Kaggle close.
- Whether the September leaders' public-LB gains transfer to the private half (public/private is a 50/50 split of windows per last year's rules; a timeline-reconstruction method transfers fully because it uses all test windows jointly).
- The request text contained a GitHub personal access token; it was not used (credentials are never entered by this agent) and should be rotated since it was pasted into a prompt.

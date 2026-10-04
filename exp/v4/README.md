# exp/v4 — multi-fit decoding of the "Learned Links + Counts" v4 pipeline

The public notebook by goodpjw2008 (v4, built on Briano / Hanh Tran / Jiwei Liu, Apache 2.0) is forked on Kaggle
(`kaggle/v4_fork/`) with different seeds; every fork keeps the inputs of its decode stage (`keep4/`). Everything here
runs on CPU from those arrays.

| script | what it does |
|---|---|
| `v4_local.py` | Re-implementation of the kernel's stage-B decode (kNN smoothing g, bagged L3 links, learned count prior with nested 5-fold ridge, Sinkhorn) and of its boundary refiner. With one fit and the kernel's settings it reproduces the kernel exactly (OOF 0.9230, test agreement 1.0000 on the 10-03 fork). Options: `--whiten 0.5 --whiten_mode subject` (per-subject whitened kNN embedding), `--onehot` (exercise identity in the count regressor), `--links` (replacement link sets), joint multi-fit mode. |
| `v4_combine.py` | Late fusion: geometric mean of every fit's final graph probabilities, count prior re-estimated on the fused P, Sinkhorn, refiner along all fits' matchings. Accepts keep4 dirs, local decodes, and Q-only parts (`q:<final_probabilities.npz>`). |
| `relink.py` | Link-level fusion: L3 link-scorer log-odds averaged over fits (union of candidate lists), then the kernel's Hungarian assignment, perturbed matchings and quantile normalisation. |

## Findings (2026-10-04)

* **Per-subject whitening of the kNN embedding.** The notebook's optional whitening projects every subject on the
  principal axes of the *training* tiles. In the out-of-fold evaluation those axes include the evaluated subject; the
  test subjects (new locations, another camera) never get that. Whitening every subject with its *own* covariance
  (label-free, shrunk 0.1 towards a scaled identity) treats training and test subjects the same way.
  Single 10-03 fit: OOF 0.9255 (no whitening) -> 0.9287 (training axes, LB 0.92513) -> 0.9296 (per subject, LB 0.92758).
* **Exercise identity in the count regressor**: count error 7.5 -> 7.1 tiles, OOF +0.0012.
* **kNN smoothing 1.5** (notebook v4 setting) on the 10-03 fits: +0.0009 / +0.0013.
* Late fusion of the two 10-03 fits, each per-subject whitened: OOF 0.9298, LB 0.92880; adding the public v4 run's
  final Q (as P proxy): OOF 0.9304, LB 0.92847.
* Flat (plateau): whitening power 0.5/0.7, shrinkage 0.1/0.3, kNN temperature 0.2/0.3/0.45, Sinkhorn sharpening 0.4-0.6.

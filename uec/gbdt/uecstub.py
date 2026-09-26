"""Make the UEC repo's train_inertial_gbdt importable here without touching the repo.

- adds E:\\Claude code\\wear\\uec\\repo to sys.path (approach_base is a namespace package)
- stubs the helper modules that are missing from the published repo
  (approach_base.src.visualize_confusion_matrix, approach_base.src.output_utils)
The feature code itself (make_feature_vector, build_record_table, ...) is used verbatim.
"""
import os, sys, types

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = r"E:\Claude code\wear\uec\repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _install_stubs():
    if "approach_base.src.visualize_confusion_matrix" not in sys.modules:
        m = types.ModuleType("approach_base.src.visualize_confusion_matrix")
        m.plot_confusion_matrix = lambda *a, **k: None
        m.reorder_confusion_matrix_for_blocks = lambda cm, labels: (cm, labels, None)
        sys.modules[m.__name__] = m
    if "approach_base.src.output_utils" not in sys.modules:
        m = types.ModuleType("approach_base.src.output_utils")
        m.make_experiment_dir = lambda base, name=None: base
        m.save_run_metadata = lambda *a, **k: None
        sys.modules[m.__name__] = m


_install_stubs()
import approach_base.src.train_inertial_gbdt as tig  # noqa: E402

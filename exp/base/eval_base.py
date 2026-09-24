import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *

t = time.time()
o1 = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy")); o2 = np.load(os.path.join(WORK, "fusion_v1", "oof.npy"))
print("nan frac", np.isnan(o1[:, :, 0]).mean(), np.isnan(o2[:, :, 0]).mean(), "both nan same", (np.isnan(o1[:, :, 0]) == np.isnan(o2[:, :, 0])).mean())
eval_single(o1, name="lgbm_v1"); eval_single(o2, name="fusion_v1")
b = blend([o1, o2], [0.8, 0.2]); eval_single(b, name="blend 0.8/0.2")
os.makedirs(os.path.join(EXP, "blend_v1f"), exist_ok=True); np.save(os.path.join(EXP, "blend_v1f", "oof.npy"), b)
if len(sys.argv) > 1:
    res, out = run_sim(os.path.join(EXP, "blend_v1f", "oof.npy"), "blend_v1f"); print(out[-1500:]); print(res)
print(time.time() - t)

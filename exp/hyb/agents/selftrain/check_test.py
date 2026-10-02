import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
bl = np.load(os.path.join(KEEP, "blend.npz"))
kb = np.load(os.path.join(KEEP, "test_logp_b.npy")).astype(np.float64)
rec = H.tab_blend(bl["test_logp"].astype(np.float64), [(bl["tab_S3"].astype(np.float64), 0.5), (bl["tab_T"].astype(np.float64), 0.3)])
print("kernel tab_S3 range", bl["tab_S3"].min(), bl["tab_S3"].max(), "rowsum exp", np.exp(bl["tab_S3"]).sum(1)[:3])
print("keep test_logp_b vs lsm(.2 win+.5 tab_S3+.3 tab_T): max abs", np.abs(kb - rec).max(), "argmax agree", np.mean(kb.argmax(1) == rec.argmax(1)))
lb = np.load(os.path.join(HB, "test_base_test_logp_b.npy"))
print("local rerun tab blend vs keep: argmax agree", np.mean(lb.argmax(1) == kb.argmax(1)))
ts3 = np.load(os.path.join(HB, "hb_S3.npy")) if os.path.exists(os.path.join(HB, "hb_S3.npy")) else None
if ts3 is not None:
    print("hb_S3 shape", ts3.shape, "agree with tab_S3 argmax", np.mean(ts3.argmax(1) == bl["tab_S3"].argmax(1)))
best = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy()
print("best sub vs keep tab argmax", np.mean(best == kb.argmax(1)), "vs local tab argmax", np.mean(best == lb.argmax(1)))
l2 = np.load(os.path.join(KEEP, "links_L2_test.npz")); su = l2["succ"]; print("L2 coverage", np.mean(su >= 0), "indeg>1", np.sum(np.bincount(su[su >= 0], minlength=len(su)) > 1))

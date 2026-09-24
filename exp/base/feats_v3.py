"""Feature blocks for the v3 booster (cached in exp/base/cache/).
IMU block (per second, per limb): limb_features on MIRROR-CANONICALISED signals (right arm: -x, right leg: -y, so left/right
share one frame) = 110 raw + 110 z-scored per (session|test-subject, limb).
VIDEO block (per second): raw mean(160), per-session/test-subject centred mean(160), std of first 64 comps, last3-first3 delta
(48), temporal DCT k=1..3 of first 24 comps (72), mean |frame delta| of first 32 comps (32) + total motion (1).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
sys.path.insert(0, r"E:\Claude code\wear\src")
from imu_feats import limb_features

CACHE = os.path.join(EXP, "cache"); os.makedirs(CACHE, exist_ok=True)
MIRROR_AXIS = {0: None, 1: None, 2: 0, 3: 1}   # right_arm negate x, right_leg negate y

def canon(W, li):
    W = np.asarray(W, np.float32).copy()
    ax = MIRROR_AXIS[li]
    if ax is not None: W[:, :, ax] *= -1
    return W

def zscore_groups(F, groups):
    F = F.copy()
    for g in np.unique(groups):
        m = groups == g
        mu = np.nanmean(F[m], 0); sd = np.nanstd(F[m], 0)
        F[m] = (F[m] - mu) / (sd + 1e-3)
    return F

def centre_groups(F, groups):
    F = F.copy()
    for g in np.unique(groups):
        m = groups == g; F[m] = F[m] - np.nanmean(F[m], 0)
    return F

def dct_basis(T=15, K=(1, 2, 3)):
    t = np.arange(T); return np.stack([np.cos(np.pi * (t + 0.5) * k / T) for k in K], 0).astype(np.float32)  # (K,T)

def video_block(Z, groups):
    Z = np.asarray(Z, np.float32)
    mu = Z.mean(1)
    muc = centre_groups(mu, groups)
    sd = Z[:, :, :64].std(1)
    dl = Z[:, -3:, :48].mean(1) - Z[:, :3, :48].mean(1)
    B = dct_basis()                                       # (3,15)
    dct = np.einsum("kt,ntc->nkc", B, Z[:, :, :24]).reshape(len(Z), -1)
    fd = np.abs(np.diff(Z, axis=1))
    mot = fd[:, :, :32].mean(1); tot = np.linalg.norm(np.diff(Z, axis=1), axis=2).mean(1, keepdims=True)
    return np.concatenate([mu, muc, sd, dl, dct, mot, tot], 1).astype(np.float32)

def train_blocks():
    f = os.path.join(CACHE, "v3_train.npz")
    if os.path.exists(f):
        d = np.load(f); return d["IM"], d["VB"]
    t = time.time(); m = meta(); sess = m.session.to_numpy()
    imu = np.load(os.path.join(PREP, "train_imu.npy"))
    IM = []
    for li in range(4):
        F = limb_features(canon(imu[:, li], li)); Fz = zscore_groups(F, sess)
        IM.append(np.concatenate([F, Fz], 1)); print("imu limb", li, F.shape, f"{time.time()-t:.0f}s", flush=True)
    IM = np.stack(IM, 1); del imu
    vid = np.load(os.path.join(PREP, "train_vid_pca.npy"))
    VB = video_block(vid, sess); del vid
    print("video block", VB.shape, f"{time.time()-t:.0f}s", flush=True)
    np.savez(f, IM=IM, VB=VB); return IM, VB

def test_blocks():
    f = os.path.join(CACHE, "v3_test.npz")
    if os.path.exists(f):
        d = np.load(f); return d["IM"], d["VB"], d["limb"]
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); sbj = tm.sbj_id.to_numpy()
    F = np.zeros((len(tm), 110), np.float32)
    for li in range(4):
        mm = limb == li; F[mm] = limb_features(canon(xi[mm], li))
    grp = np.array([f"{s}|{l}" for s, l in zip(sbj, limb)])
    IM = np.concatenate([F, zscore_groups(F, grp)], 1)
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"))
    VB = video_block(vid, sbj)
    np.savez(f, IM=IM, VB=VB, limb=limb); return IM, VB, limb

def train_imu_zgroups_check():
    pass

if __name__ == "__main__":
    IM, VB = train_blocks(); print(IM.shape, VB.shape)
    a, b, c = test_blocks(); print(a.shape, b.shape, c.shape)

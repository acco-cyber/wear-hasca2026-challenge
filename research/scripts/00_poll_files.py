"""Poll until test_videomae_data.npy has exact expected size and train/inertial_feat/sbj_0.csv exists (max 25 min)."""
import os, time, sys
V = r"E:\Claude code\wear\data\test\test_videomae_data.npy"
T = r"E:\Claude code\wear\data\train\inertial_feat\sbj_0.csv"
EXP = 1127485568
t0 = time.time()
while time.time() - t0 < 25 * 60:
    vs = os.path.getsize(V) if os.path.exists(V) else -1
    te = os.path.exists(T) and os.path.getsize(T) > 0
    print(f"[{time.time()-t0:5.0f}s] video size={vs} ({vs/EXP*100 if vs>0 else 0:.1f}%) train_sbj0={te}", flush=True)
    if vs == EXP and te:
        print("READY"); sys.exit(0)
    time.sleep(30)
print("TIMEOUT"); sys.exit(1)

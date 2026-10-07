import os, csv, numpy as np
W = r"E:\Claude code\wear"; K7=os.path.join(W,"work","v4","wear-v4-big-pool-opt-s7","keep4")
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
st=np.load(os.path.join(K7,"stage.npz"),allow_pickle=True); sens=st["sensor_test"]; sb=st["test_sbj"]
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
lk=np.load(os.path.join(K7,"links.npz")); su=lk["test_succ"][0]; sc=lk["test_score"][0]
for L in (0,3):
    i=np.flatnonzero((sens==L)&(su>=0)); j=su[i]; same=sens[j]==L; hi=sc[i]>np.quantile(sc[su>=0],0.5)
    k=i[same&hi]; jj=su[k]
    print(L,"base ex",ex[sens==L].mean().round(3),"P(ex_j|ex_i)",ex[jj][ex[k]].mean().round(3),"P(ex_j|~ex_i)",ex[jj][~ex[k]].mean().round(3), len(k))
    # cross-limb: does exact status of arm tile correlate with time over 2-step paths
# noise level: 2nd diff power of 2025 rows by limb, twins exact vs non-exact
def d2(X): return (np.diff(X,2,axis=1)**2).sum(2).mean(1)
for l in range(4):
    r=np.flatnonzero(l25==l); print("w25 limb",l,"d2 quantiles",np.quantile(d2(A25[r]),[0.05,0.25,0.5]).round(4))
for L in range(4):
    k=np.flatnonzero(sens==L); print("ours sens",L,"d2 q",np.quantile(d2(A26[k]),[0.05,0.25,0.5]).round(4), "twin ex",np.quantile(d2(A25[twin[k[ex[k]]]]),[0.05,0.25,0.5]).round(4) if ex[k].any() else "", "twin non-ex",np.quantile(d2(A25[twin[k[~ex[k]]]]),[0.05,0.25,0.5]).round(4) if (~ex[k]).any() else "")

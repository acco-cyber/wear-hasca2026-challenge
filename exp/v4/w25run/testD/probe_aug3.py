import os, csv, numpy as np
W = r"E:\Claude code\wear"; K7=os.path.join(W,"work","v4","wear-v4-big-pool-opt-s7","keep4")
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]; id25=z["id"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
st=np.load(os.path.join(K7,"stage.npz"),allow_pickle=True); sens=st["sensor_test"]; sb=st["test_sbj"]
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
for L in (0,3):
    k=sens==L
    print(L, "ex vs id26 parity", [ex[k&(np.arange(len(ex))%2==p)].mean().round(3) for p in (0,1)], "ex vs twin id parity", [ex[k&(id25[twin]%2==p)].mean().round(3) for p in (0,1)])
    print("   ex vs twin id mod 4", [ex[k&(id25[twin]%4==p)].mean().round(3) for p in range(4)], "ex vs id26 mod 4",[ex[k&(np.arange(len(ex))%4==p)].mean().round(3) for p in range(4)])
# all K7 same-limb link transitions by score bin
lk=np.load(os.path.join(K7,"links.npz")); su=lk["test_succ"][0]; sc=lk["test_score"][0]
for L in (0,3):
    i=np.flatnonzero((sens==L)&(su>=0)); j=su[i]; i=i[sens[j]==L]; j=su[i]
    print(L,"all same-limb links", len(i), "EE",np.mean(ex[i]&ex[j]).round(3),"EN",np.mean(ex[i]&~ex[j]).round(3),"NE",np.mean(~ex[i]&ex[j]).round(3),"NN",np.mean(~ex[i]&~ex[j]).round(3))
# 2-step: for chains of length via links regardless of limb: ex status of arms around
# check d2 (noise) of ours for N vs E tiles
def d2(X): return (np.diff(X,2,axis=1)**2).sum(2).mean(1)
for L in (0,3):
    k=sens==L; print(L,"ours d2 median E",np.median(d2(A26[k&ex])).round(5),"N",np.median(d2(A26[k&~ex])).round(5), "std E", np.median(A26[k&ex].std(1).sum(1)).round(4), "N", np.median(A26[k&~ex].std(1).sum(1)).round(4))

import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]
sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
def zs(M):
    M=M-M.mean(1,keepdims=True); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
m25=np.linalg.norm(A25,axis=2); m26=np.linalg.norm(A26,axis=2)
used=np.zeros(len(A25),bool); used[twin[ex]]=True
for s in (22,23):
    r=np.flatnonzero((s25==s)&~used)
    for L in (3,0):
        k=np.flatnonzero(~ex&(sens26==L)&(s26==s)); act=A26[k].std(1).sum(1); k=k[act>np.quantile(act,0.5)]
        C=zs(m26[k])@zs(m25[r]).T; b=C.argmax(1); top=C.max(1)
        print(s, SENS[L], len(k), "best corr over all unused rows q", np.quantile(top,[0.1,0.5,0.9]).round(4), "limb of best", np.unique(pl25[r[b]],return_counts=True), flush=True)
        # also: best over same-limb unused rows, with time shifts of +-1..10 samples (magnitude cross-corr on overlap)
        rl=r[pl25[r]==L]; bests=[]
        for sh in (-10,-5,-2,-1,0,1,2,5,10):
            if sh>=0: a=m26[k][:,sh:]; bb=m25[rl][:,:50-sh]
            else: a=m26[k][:,:50+sh]; bb=m25[rl][:,-sh:]
            Cs=zs(a)@zs(bb).T; bests.append(Cs.max(1))
        bests=np.array(bests); print("   same-limb shift best corr median per shift", dict(zip((-10,-5,-2,-1,0,1,2,5,10), np.median(bests,1).round(3))), flush=True)

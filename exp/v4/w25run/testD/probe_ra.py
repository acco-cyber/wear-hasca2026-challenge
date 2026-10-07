import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin=m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]
sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
conf=~ex&(corr>0.99)&(margin>0.05)
used=np.zeros(len(A25),bool); used[twin[ex]]=True
# Procrustes R from confident RA pairs (pooled)
k=np.flatnonzero(conf&(sens26==0))
Xs=np.concatenate([A26[i] for i in k]); Bs=np.concatenate([A25[twin[i]] for i in k])
U,_,Vt=np.linalg.svd(Xs.T@Bs); R=U@Vt; print("pooled Procrustes R\n",R.round(3)," det",np.linalg.det(R).round(3))
res=[(A25[twin[i]]-A26[i]@R).std() for i in k]; print("residual std q",np.quantile(res,[0.1,0.5,0.9]).round(4))
# per-row noise estimate of all RA 2025 rows: d2
def d2(X): return (np.diff(X,2,axis=1)**2).sum(2).mean(1)
for s in (22,23,24,25):
    r=np.flatnonzero((s25==s)&(pl25==0)&~used)
    q=np.flatnonzero(~ex&(sens26==0)&(s26==s))
    XR=np.einsum("ntc,cd->ntd",A26[q],R)
    D=((XR[:,None]-A25[r][None])**2).mean((2,3))**0.5 if len(q)*len(r)<4e6 else None
    if D is None:
        D=np.stack([((XR[i][None]-A25[r])**2).mean((1,2))**0.5 for i in range(len(q))])
    b=D.argmin(1); best=D.min(1); D2=D.copy(); D2[np.arange(len(q)),b]=9; sec=D2.min(1)
    print(s,"RA non-exact",len(q),"unused rows",len(r),"best rms q",np.quantile(best,[0.1,0.5,0.9,0.95]).round(3),"second q",np.quantile(sec,[0.1,0.5]).round(3),
          "share best<0.13 & sec>1.5*best",np.mean((best<0.13)&(sec>1.5*best)).round(3),"unique",len(np.unique(b[(best<0.13)])), (best<0.13).sum(), flush=True)
    # noise: d2 of the unused rows
    dd=d2(A25[r]); print("   unused RA rows d2 q",np.quantile(dd,[0.1,0.3,0.5,0.7,0.9]).round(4), " twins-best rows d2 med", np.median(d2(A25[r[b]])).round(4))

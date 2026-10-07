import os, csv, numpy as np
W = r"E:\Claude code\wear"; C=r"E:\Claude code\wear\exp\v4\w25run\testD\cache"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25=z["sbj"]; l25=z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
d=np.load(os.path.join(C,"deaug_inputs.npz")); twin=d["twin"]; how=d["how"]; augRA=d["augRA"]
used=np.zeros(len(A25),bool); used[twin[twin>=0]]=True; used|=augRA
cands=[]
for s in (22,23,24,25):
    r=np.flatnonzero((s25==s)&(pl25==0)&~used); q=np.flatnonzero((how==0)&(sens26==0)&(s26==s))
    act=A26[q].std(1).sum(1); q=q[act>np.quantile(act,0.5)]
    B=A25[r]; Bc=B-B.mean(1,keepdims=True); den=(Bc**2).sum((1,2)); BB=(B**2).sum((1,2))
    for i in q:
        X=np.c_[A26[i],np.ones(50)]; Q,_=np.linalg.qr(X)
        proj=np.einsum("tk,mtc->mkc",Q,B); ratio=(BB-(proj**2).sum((1,2)))/den
        o=np.argsort(ratio)[:2]
        if ratio[o[0]]<0.08 and ratio[o[1]]>2*ratio[o[0]]: cands.append((i,r[o[0]],ratio[o[0]]))
print("candidate pairs", len(cands))
Xq=np.stack([A26[i] for i,_,_ in cands]); Bq=np.stack([A25[j] for _,j,_ in cands])
keep=np.ones(len(cands),bool)
for rnd in range(6):
    A_=np.linalg.lstsq(Xq[keep].reshape(-1,3),Bq[keep].reshape(-1,3),rcond=None)[0]
    for it in range(20):
        N=(Bq[keep]-Xq[keep]@A_).mean(0); A_=np.linalg.lstsq(Xq[keep].reshape(-1,3),(Bq[keep]-N[None]).reshape(-1,3),rcond=None)[0]
    res=np.sqrt(((Bq-Xq@A_-N[None])**2).mean((1,2)))
    print("round",rnd,"kept",keep.sum(),"res q",np.quantile(res,[0.1,0.25,0.5]).round(4)); keep=res<max(np.quantile(res[keep],0.5),1e-3)
print(A_.round(4), "det", np.linalg.det(A_).round(4), "N rms", np.sqrt((N**2).mean()).round(4))
np.savez(os.path.join(C,"typeii_probe.npz"),A=A_,N=N)

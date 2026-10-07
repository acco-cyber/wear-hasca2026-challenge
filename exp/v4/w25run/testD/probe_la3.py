import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
used=np.zeros(len(A25),bool); used[twin[ex]]=True
s=24; r=np.flatnonzero((s25==s)&(pl25==3)&~used); q=np.flatnonzero(~ex&(sens26==3)&(s26==s))
act=A26[q].std(1).sum(1); q=q[np.argsort(-act)][:40]
for i in q[:6]:
    x=-A26[i]; D=np.sqrt(((x[None]-A25[r])**2).mean((1,2))); j=r[D.argmin()]; b=A25[j]
    d=b-x
    Xa=np.c_[A26[i],np.ones(50)]; Rf=np.linalg.lstsq(Xa,b,rcond=None)[0]; res=(b-Xa@Rf).std()
    print("i",i,"rms(-x)",D.min().round(3),"2nd",np.sort(D)[1].round(3),"lin res",res.round(4)); print(Rf.round(3))
    print("  diff first 8 samples x-axis", d[:8,0].round(3), " ratio b/x mean", (np.abs(b).sum()/np.abs(x).sum()).round(3))

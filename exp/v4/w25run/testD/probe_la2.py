import os, csv, itertools, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
used=np.zeros(len(A25),bool); used[twin[ex]]=True
for P,name in ((-np.eye(3),"neg"), (np.array([[1,0,0],[0,0,1],[0,1,0]])*np.array([-1,1,1])[None,:],"p021s-11")):
  for s in (22,23,24,25):
    r=np.flatnonzero((s25==s)&(pl25==3)&~used); q=np.flatnonzero(~ex&(sens26==3)&(s26==s))
    XP=A26[q]@P
    D=np.stack([np.abs(XP[i][None]-A25[r]).max((1,2)) for i in range(len(q))])
    b=D.min(1); print(name, s, len(q), "best maxdiff q", np.quantile(b,[0.1,0.5,0.9]).round(4), "share <1e-4", np.mean(b<1e-4).round(3), "rms-best q", np.quantile(np.sqrt(np.stack([((XP[i][None]-A25[r])**2).mean((1,2)) for i in range(min(len(q),150))]).min(1)),[0.1,0.5,0.9]).round(3), flush=True)

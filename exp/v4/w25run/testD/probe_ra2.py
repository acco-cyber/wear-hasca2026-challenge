import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr=m["corr"]; margin=m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
conf=~ex&(corr>0.99)&(margin>0.05)&(sens26==0)
k=np.flatnonzero(conf)
Xs=np.concatenate([A26[i] for i in k]); Bs=np.concatenate([A25[twin[i]] for i in k]); U,_,Vt=np.linalg.svd(Xs.T@Bs); R=U@Vt
E=np.stack([A25[twin[i]]-A26[i]@R for i in k])   # (n,50,3)
print("RA residual std per sample index (first 5, mid, last 5):", E.std((0,2))[:5].round(4), E.std((0,2))[24:26].round(4), E.std((0,2))[-5:].round(4))
print("residual std per axis", E.std((0,1)).round(4), "mean", E.mean((0,1)).round(4))
# lag-1 autocorrelation of residual (white noise?)
print("resid lag1 autocorr", np.mean(E[:,1:]*E[:,:-1])/np.mean(E**2))
# is R exactly a signed permutation-like in some frame? print R in full precision
print(R.round(4))
# try: is b = (x + noise) @ R or with time-warp? check residual vs local slope
sl=np.gradient(np.stack([A26[i]@R for i in k]),axis=1); print("corr(resid, slope)", np.corrcoef(E.ravel(), sl.ravel())[0,1].round(4))

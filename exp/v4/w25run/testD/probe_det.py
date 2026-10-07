import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr=m["corr"]; margin=m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
used=np.zeros(len(A25),bool); used[twin[ex]]=True
# LA: b0==b1 detector
for L,name in ((3,"LA"),(0,"RA")):
    r=np.flatnonzero(pl25==L)
    eq01=np.abs(A25[r][:,0]-A25[r][:,1]).max(1)<1e-7
    tw_ex=twin[ex&(sens26==L)]; tw_ne=twin[~ex&(sens26==L)]
    e=lambda rr: (np.abs(A25[rr][:,0]-A25[rr][:,1]).max(1)<1e-7).mean().round(4)
    print(name,"2025 rows b0==b1 share all",eq01.mean().round(4),"| exact-twin rows",e(tw_ex),"| non-exact twin rows",e(tw_ne), "| ours", (np.abs(A26[sens26==L][:,0]-A26[sens26==L][:,1]).max(1)<1e-7).mean().round(4))
# RA: per-tile rotation residual whiteness on confident pairs
k=np.flatnonzero(~ex&(corr>0.99)&(margin>0.05)&(sens26==0))
ac=[];e0=[];em=[]
for i in k:
    x=A26[i]; b=A25[twin[i]]; R=np.linalg.lstsq(x,b,rcond=None)[0]; E=b-x@R
    ac.append(np.mean(E[1:]*E[:-1])/np.mean(E**2)); e0.append(np.abs(E[0]).max()); em.append(np.abs(E[25]).max())
print("RA resid lag1 autocorr median", np.median(ac).round(3), "|E[0]| med", np.median(e0).round(3), "|E[25]| med", np.median(em).round(3))

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
f25=A25[r][:,0,:]
taus=[]
grid=np.linspace(0,49,49*20+1)
for i in q:
    k=np.flatnonzero(np.abs(f25+A26[i][0][None]).max(1)<1e-5)[0]; b=-A25[r[k]]; x=A26[i]
    if x.std(0).sum()<0.5: continue
    xi=np.stack([np.interp(grid,np.arange(50),x[:,c]) for c in range(3)],1)   # (G,3)
    tau=[]
    prev=0
    for t in range(50):
        lo=max(0,prev-60); hi=min(len(grid),prev+200)   # monotone search window
        d=((xi[lo:hi]-b[t][None])**2).sum(1); j=lo+d.argmin(); tau.append(grid[j]); prev=j
    taus.append(np.array(tau))
    if len(taus)>=8: break
for t in taus: print(np.round(t[[0,1,2,3,5,10,20,25,30,40,45,47,48,49]]-np.array([0,1,2,3,5,10,20,25,30,40,45,47,48,49]),2))

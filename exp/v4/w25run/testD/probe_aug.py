import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); loc26 = np.array([r["sensor_location"] for r in rows])
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; L25 = ["left_arm", "left_leg", "right_arm", "right_leg"]
sens26=np.array([SENS.index(x) for x in loc26]); map25=np.array([SENS.index(n) for n in L25]); pl25=map25[l25]
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
conf=~ex&(corr>0.99)&(margin>0.05)
print("A26 dtype", np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).dtype)
for s in (22,23,24,25):
  for L in (0,3):
    k=np.flatnonzero(conf&(s26==s)&(sens26==L))
    Rs=[]; rs=[]
    for i in k:
        a,b=A26[i],A25[twin[i]]; R=np.linalg.lstsq(a,b,rcond=None)[0]; Rs.append(R); rs.append((b-a@R).std())
    if not len(k): print(s,L,"none"); continue
    Rs=np.array(Rs); print(s,SENS[L],len(k),"R mean\n",Rs.mean(0).round(3),"\n R std",Rs.std(0).max().round(3),"res std med",np.median(rs).round(4))
    # orthogonality
    Rm=Rs.mean(0); print(" RtR", (Rm.T@Rm).round(3).tolist())

import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]
sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
for L in (3,0):
    k=np.flatnonzero(~ex&(sens26==L)); print(SENS[L], "non-exact", len(k), "corr q", np.quantile(corr[k],[0.1,0.5,0.9]).round(3))
    # high-activity ones
    act=A26[k].std(1).sum(1); kk=k[np.argsort(-act)[:300]]
    print("  top-activity corr q", np.quantile(corr[kk],[0.1,0.5,0.9]).round(3))
    # for best of them, compare x and b
    for i in kk[:3]:
        x=A26[i]; b=A25[twin[i]]
        Xa=np.c_[x,np.ones(50)]; R=np.linalg.lstsq(Xa,b,rcond=None)[0]; res=(b-Xa@R).std()
        print("  i",i,"corr",corr[i].round(3),"lin res",res.round(3),"\n", R.round(2))
        print("   x mean", x.mean(0).round(2), "b mean", b.mean(0).round(2), "x std", x.std(0).round(2), "b std", b.std(0).round(2))
    # remaining (non-twin) 2025 rows of that limb: stats of the mean vector (gravity direction) for exact twins vs others
    r=np.flatnonzero(pl25==L); used=np.zeros(len(A25),bool); used[twin[ex]]=True
    mm=A25[r].mean(1); print("  2025 rows mean-vector quantiles x/y/z (all rows)", np.quantile(mm,[0.1,0.5,0.9],axis=0).round(2).tolist())
    mo=A26[sens26==L].mean(1); print("  ours mean-vector quantiles", np.quantile(mo,[0.1,0.5,0.9],axis=0).round(2).tolist())
    d2=lambda X:(np.diff(X,2,axis=1)**2).sum(2).mean(1)
    print("  d2 2025 rows q", np.quantile(d2(A25[r]),[0.25,0.5,0.75]).round(4), "ours", np.quantile(d2(A26[sens26==L]),[0.25,0.5,0.75]).round(4))

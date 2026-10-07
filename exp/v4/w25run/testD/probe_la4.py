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
for s in (22,23,24,25):
    r=np.flatnonzero((s25==s)&(pl25==3)&~used); q=np.flatnonzero(~ex&(sens26==3)&(s26==s))
    # match on first sample exactly (negated)
    f25=A25[r][:,0,:]; hits=[]; lastd=[]; nh=0
    for i in q:
        d=np.abs(f25+A26[i][0][None]).max(1); k=np.flatnonzero(d<1e-5)
        hits.append(len(k))
        if len(k)==1:
            b=A25[r[k[0]]]; lastd.append(np.abs(b[-1]+A26[i][-1]).max()); 
    hits=np.array(hits); lastd=np.array(lastd)
    print(s,"LA non-exact",len(q),"first-sample(-x) exact hits: share>=1",np.mean(hits>=1).round(3),"unique",np.mean(hits==1).round(3),"last-sample |diff| q",np.quantile(lastd,[0.1,0.5,0.9]).round(4) if len(lastd) else None, flush=True)
    # positive (non negated) first sample?
    hp=[len(np.flatnonzero(np.abs(f25-A26[i][0][None]).max(1)<1e-5)) for i in q]; print("   first-sample(+x) hits share", np.mean(np.array(hp)>=1).round(3))

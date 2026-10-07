import os, csv, numpy as np
W = r"E:\Claude code\wear"
K7=os.path.join(W,"work","v4","wear-v4-big-pool-opt-s7","keep4")
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; id25 = z["id"]; s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
id26 = np.array([int(r["id"]) for r in rows]); s26 = np.array([int(r["sbj_id"]) for r in rows]); loc26 = np.array([r["sensor_location"] for r in rows])
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; L25 = ["left_arm", "left_leg", "right_arm", "right_leg"]
st=np.load(os.path.join(K7,"stage.npz"),allow_pickle=True)
ids=np.array([int(x) for x in st["ids"]]); print("pipeline ids == arange", (ids==np.arange(len(ids))).all(), "set eq", (np.sort(ids)==np.sort(id26)).all())
sens26=np.array([SENS.index(x) for x in loc26])
print("sens agree", (st["sensor_test"]==sens26[ids]).mean(), "sbj agree", (st["test_sbj"]==s26[ids]).mean())
diff = np.abs(A26.astype(np.float64) - A25[twin].astype(np.float64)).max((1, 2))
for th in (1e-6,1e-4,1e-3,1e-2):
    print(th, (diff<th).mean())
ex=diff<1e-4
map25=np.array([SENS.index(n) for n in L25]); print("map25", map25)
print("exact twin limb agree", (map25[l25[twin[ex]]]==sens26[ex]).mean(), "sbj", (s25[twin[ex]]==s26[ex]).mean())
print("unique twins all", len(np.unique(twin)), "exact", len(np.unique(twin[ex])), ex.sum())
ne=~ex
print("non-exact corr q", np.quantile(corr[ne],[0.01,0.05,0.1,0.25,0.5]).round(4), "margin q", np.quantile(margin[ne],[0.01,0.05,0.1,0.25,0.5]).round(4))
conf=ne&(corr>0.99)&(margin>0.05); anch=ex|conf
print("conf non-exact", conf.sum(), conf.mean(), "anchored", anch.mean(), "unanchored", (~anch).sum())
print("unique anchored twins", len(np.unique(twin[anch])), anch.sum())
# duplicates among twins
u,c=np.unique(twin,return_counts=True); print("dup twin rows", (c>1).sum())
# among exact twins, per (sbj, limb) share
for s in np.unique(s26):
    print(s, [ (round(ex[(s26==s)&(sens26==L)].mean(),3), round(anch[(s26==s)&(sens26==L)].mean(),3)) for L in range(4)])
# exact matches among all 2025 rows: hash
from collections import defaultdict
h25={}
for i in range(len(A25)): h25.setdefault(A25[i].tobytes(),[]).append(i)
hit=np.array([len(h25.get(A26[i].tobytes(),[])) for i in range(len(A26))])
print("raw-byte exact hits share", (hit>0).mean(), "multi", (hit>1).sum())
first=np.array([h25[A26[i].tobytes()][0] if hit[i] else -1 for i in range(len(A26))])
k=hit==1; print("hash-match limb agree", (map25[l25[first[k]]]==sens26[k]).mean(), "== twin", (first[k]==twin[k]).mean())
# augmentation probe on non-exact confident: fit linear map
res=[];sc=[]
for i in np.flatnonzero(conf)[:500]:
    a,b=A26[i].astype(np.float64),A25[twin[i]].astype(np.float64); R=np.linalg.lstsq(a,b,rcond=None)[0]; res.append(np.abs(b-a@R).max()); sc.append(np.round(R,2))
print("linear fit maxres q", np.quantile(res,[0.5,0.9,0.99]))
for r in sc[:8]: print(r)

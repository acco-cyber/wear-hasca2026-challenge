import os, csv, itertools, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; sens26=np.array([SENS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2))<1e-4
MAP25=np.array([3,2,0,1]); pl25=MAP25[l25]
used=np.zeros(len(A25),bool); used[twin[ex]]=True
rng=np.random.default_rng(0)
def ed(Aa,Bb):
    a=Aa[rng.choice(len(Aa),min(1500,len(Aa)),replace=False)]; b=Bb[rng.choice(len(Bb),min(1500,len(Bb)),replace=False)]
    d=lambda x,y: np.linalg.norm(x[:,None]-y[None],axis=2).mean()
    return 2*d(a,b)-d(a,a)-d(b,b)
for L in (3,0):
    # unused 2025 rows of limb L that are NOT exact twins; among them clean ones look like ours. Mixture: compare features of non-twin rows
    r=np.flatnonzero((pl25==L)&~used)
    feat=lambda X: np.c_[X.mean(1), X.std(1)]
    F25=feat(A25[r]); ours=A26[sens26==L]; pi_aug = 1-ex[sens26==L].mean(); share_aug_unused = pi_aug/(1-ex[sens26==L].mean()*0.25)
    print(SENS[L], "aug share", round(pi_aug,3), "expected aug share among unused rows", round(share_aug_unused,3), flush=True)
    res=[]
    cands={}
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1,-1],repeat=3):
            P=np.zeros((3,3)); P[list(range(3)),list(perm)]=1; P=P*np.array(sg)[None,:]; cands[f"p{perm}s{sg}"]=P
    U=np.array([[0.342,-0.474,-0.812],[0.819,-0.274,0.505],[0.461,0.837,-0.294]]); cands["R_RA"]=U; cands["R_RA_T"]=U.T
    for k,P in cands.items():
        n_aug=int(share_aug_unused*2000); idx=rng.choice(len(ours),2000)
        Xm=ours[idx].copy(); Xm[:n_aug]=Xm[:n_aug]@P
        if L==0: Xm[:n_aug]+=rng.normal(0,0.104,Xm[:n_aug].shape)
        res.append((ed(feat(Xm),F25),k))
    res.sort(); print("  best:", [(round(v,4),k) for v,k in res[:5]], "identity:", [round(v,4) for v,k in res if k=="p(0, 1, 2)s(1, 1, 1)"], flush=True)

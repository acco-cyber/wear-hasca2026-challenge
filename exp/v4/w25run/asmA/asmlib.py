"""asmA: greedy vote assembly of limb-chain fragments into a global timeline.
Pure functions on arbitrary (permuted) node / tile ids -- nothing here looks at true time, rows order or labels.
  fragments   = maximal runs of chain links with conf >= thr (per limb chain); unanchored tiles = one-tile pseudo fragments
  votes       = candidate links i -> j (time(j) = time(i) + 1) between our tiles + sub-threshold chain links (chain gaps)
  merging     = union-find with offsets, pairs by decreasing (best - second best) offset weight, >= cmin agreeing votes,
                rejected when two of our tiles or two nodes of one limb would share a time coordinate; repeated passes
                re-aggregate the votes between the grown groups."""
import numpy as np


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -30, 30)))


def build_votes(n, msu, msc, cand, Lc, lam=0.25, topL3=3):
    """msu (M, n) subject-local successor (-1 none), msc (M, n) qn scores; cand/Lc (n, K) L3 candidates (local) ->
    distinct pairs (vi, vj, w): w = mean over members of sigmoid(score) [pair in member] + lam * sigmoid(L3) for top-3."""
    M = len(msu); I, J, Wt = [], [], []
    for k in range(M):
        m = msu[k] >= 0
        I.append(np.flatnonzero(m)); J.append(msu[k][m]); Wt.append(sig(msc[k][m]) / M)
    if lam > 0 and topL3 > 0 and cand is not None:
        valid = (cand >= 0) & (Lc > -49)
        sc = np.where(valid, Lc, -np.inf)
        o = np.argsort(-sc, 1, kind="stable")[:, :topL3]
        cj = np.take_along_axis(cand, o, 1); lj = np.take_along_axis(sc, o, 1)
        ii = np.broadcast_to(np.arange(n)[:, None], cj.shape)
        ok = np.isfinite(lj) & (cj >= 0) & (cj != ii)
        I.append(ii[ok]); J.append(cj[ok]); Wt.append(lam * sig(lj[ok]))
    I = np.concatenate(I).astype(np.int64); J = np.concatenate(J).astype(np.int64); Wt = np.concatenate(Wt)
    key = I * n + J; u, inv = np.unique(key, return_inverse=True)
    return u // n, u % n, np.bincount(inv, Wt)


class Assembler:
    def __init__(self, n, lim, anch, pos, succ, conf, thr):
        self.n = n; lim = np.asarray(lim, np.int64); anch = np.asarray(anch, bool); pos = np.asarray(pos, np.int64)
        fr = np.full((4, n), -1, np.int64); off = np.zeros((4, n), np.int64); flim, flen = [], []; nf = 0
        gap = []
        for L in range(4):
            su = np.asarray(succ[L], np.int64); cf = np.asarray(conf[L], np.float64)
            ok = (su >= 0) & (cf >= thr); nxt = np.where(ok, su, -1)
            hp = np.zeros(n, bool); hp[nxt[nxt >= 0]] = True
            nl = nxt.tolist()
            for h in np.flatnonzero(~hp).tolist():
                q = h; o = 0
                while q >= 0:
                    fr[L, q] = nf; off[L, q] = o; o += 1; q = nl[q]
                flim.append(L); flen.append(o); nf += 1
            for q in np.flatnonzero(fr[L] < 0).tolist():          # nodes on a confident cycle (should not happen)
                fr[L, q] = nf; off[L, q] = 0; flim.append(L); flen.append(1); nf += 1
            g = np.flatnonzero((su >= 0) & ~ok & (cf > 0))
            gap.append((L, g, su[g], cf[g]))
        self.n_chain_frag = nf
        ft = np.full(n, -1, np.int64); ot = np.zeros(n, np.int64)
        a = np.flatnonzero(anch); ft[a] = fr[lim[a], pos[a]]; ot[a] = off[lim[a], pos[a]]
        u = np.flatnonzero(~anch); ft[u] = nf + np.arange(len(u)); nf += len(u); flim += [-1] * len(u); flen += [0] * len(u)
        self.nf = nf; self.ft, self.ot = ft, ot; self.flim = np.array(flim); self.flen = np.array(flen)
        self.fr, self.off = fr, off
        # fragment-level chain gap votes
        ga, oa, gb, ob, gw = [], [], [], [], []
        for L, g, s2, c2 in gap:
            ga.append(fr[L, g]); oa.append(off[L, g]); gb.append(fr[L, s2]); ob.append(off[L, s2]); gw.append(c2)
        self.gap = tuple(np.concatenate(x) for x in (ga, oa, gb, ob, gw))
        # union-find state + root contents
        self.parent = list(range(nf)); self.delta = [0] * nf
        self.tiles = [dict() for _ in range(nf)]; self.nodes = [set() for _ in range(nf)]; self.size = [0] * nf
        for i in range(n):
            self.tiles[ft[i]][int(ot[i])] = i
        for f in range(self.n_chain_frag):
            L = int(self.flim[f]); self.nodes[f] = set(range(L, 4 * int(self.flen[f]), 4))
        for f in range(nf):
            self.size[f] = len(self.nodes[f]) + len(self.tiles[f])
        self.merges = 0; self.rejects = 0

    def find(self, x):
        parent, delta = self.parent, self.delta
        path = []
        while parent[x] != x:
            path.append(x); x = parent[x]
        acc = 0
        for p in reversed(path):
            acc += delta[p]; delta[p] = acc; parent[p] = x
        return x, (delta[path[0]] if path else 0)

    def find_all(self):
        R = np.empty(self.nf, np.int64); O = np.empty(self.nf, np.int64)
        for f in range(self.nf):
            r, o = self.find(f); R[f] = r; O[f] = o
        return R, O

    def try_merge(self, Ra, Rb, shift):
        """Rb frame + shift = Ra frame"""
        if self.size[Ra] < self.size[Rb]:
            big, small, sh = Rb, Ra, -shift
        else:
            big, small, sh = Ra, Rb, shift
        Tb, Ts = self.tiles[big], self.tiles[small]
        for c in Ts:
            if c + sh in Tb:
                self.rejects += 1; return False
        Nb = self.nodes[big]; shifted = [k + 4 * sh for k in self.nodes[small]]
        if not Nb.isdisjoint(shifted):
            self.rejects += 1; return False
        for c, i in Ts.items():
            Tb[c + sh] = i
        Nb.update(shifted)
        self.size[big] += self.size[small]; self.tiles[small] = None; self.nodes[small] = None
        self.parent[small] = big; self.delta[small] = sh; self.merges += 1
        return True

    def run(self, vi, vj, vw, mmin, cmin=2, passes=12, use_gap=True, log=None):
        ft, ot = self.ft, self.ot
        fa = ft[vi]; oa = ot[vi]; fb = ft[vj]; ob = ot[vj]; w = np.asarray(vw, np.float64)
        if use_gap:
            g = self.gap
            fa = np.r_[fa, g[0]]; oa = np.r_[oa, g[1]]; fb = np.r_[fb, g[2]]; ob = np.r_[ob, g[3]]; w = np.r_[w, g[4]]
        hist = []
        for p in range(passes):
            R, O = self.find_all()
            ra = R[fa]; ca = O[fa] + oa; rb = R[fb]; cb = O[fb] + ob
            m = ra != rb
            D = ca + 1 - cb                                   # rb frame + D = ra frame
            A = np.minimum(ra, rb)[m]; B = np.maximum(ra, rb)[m]; rel = np.where(ra < rb, D, -D)[m]; ww = w[m]
            if not len(A):
                break
            o = np.lexsort((rel, B, A)); A, B, rel, ww = A[o], B[o], rel[o], ww[o]
            new = np.r_[True, (A[1:] != A[:-1]) | (B[1:] != B[:-1]) | (rel[1:] != rel[:-1])]
            st = np.flatnonzero(new); Wsum = np.add.reduceat(ww, st); Cnt = np.diff(np.r_[st, len(A)])
            tA, tB, tR = A[st], B[st], rel[st]
            # per pair (A, B): best and second best offset
            pn = np.r_[True, (tA[1:] != tA[:-1]) | (tB[1:] != tB[:-1])]; ps = np.flatnonzero(pn)
            pid = np.cumsum(pn) - 1
            o2 = np.lexsort((-Wsum, pid)); pid2 = pid[o2]
            first = np.r_[True, pid2[1:] != pid2[:-1]]
            best = o2[first]                                  # index into triples, one per pair (ordered by pid)
            W1 = Wsum[best]; C1 = Cnt[best]
            sec = np.zeros(len(ps))
            second_mask = np.r_[False, (pid2[1:] == pid2[:-1]) & first[:-1]]
            sec[pid2[second_mask]] = Wsum[o2[second_mask]]
            margin = W1 - sec
            sel = np.flatnonzero((C1 >= cmin) & (margin >= mmin))
            sel = sel[np.argsort(-margin[sel], kind="stable")]
            mg0 = self.merges; rj0 = self.rejects
            for k in sel.tolist():
                t = best[k]
                Ra, oa_ = self.find(int(tA[t])); Rb, ob_ = self.find(int(tB[t]))
                if Ra == Rb:
                    continue
                self.try_merge(Ra, Rb, int(tR[t]) + oa_ - ob_)
            hist.append((p, len(sel), self.merges - mg0, self.rejects - rj0))
            if log:
                log(f"pass {p}: candidate pairs {len(sel)} merges {self.merges - mg0} rejects {self.rejects - rj0}")
            if self.merges == mg0:
                break
        return hist

    def result(self):
        """per tile: assembly successor (-1), root, coordinate, tiles in group, fragments in group"""
        R, O = self.find_all(); n = self.n
        root = R[self.ft]; coord = O[self.ft] + self.ot
        gt = np.bincount(root, minlength=self.nf); gf = np.bincount(R, minlength=self.nf)
        c0 = coord - coord.min() + 1; span = int(c0.max()) + 3
        key = root * span + c0
        o = np.argsort(key); sk = key[o]
        q = key + 1; ix = np.clip(np.searchsorted(sk, q), 0, n - 1); hit = sk[ix] == q
        asu = np.where(hit, o[ix], -1)
        assert len(np.unique(key)) == n
        return dict(asu=asu, root=root, coord=coord, gtiles=gt[root], gfrags=gf[root])


def compose(msu, msc, res, gmin, hi, drop_contra=True):
    """8-member output: assembly successor (score hi) for tiles in groups with >= gmin tiles; elsewhere member k's link,
    dropped when its target already has an assembly predecessor or it contradicts the assembly (same group, wrong coord)."""
    M, n = msu.shape
    asu, root, coord, gt = res["asu"], res["root"], res["coord"], res["gtiles"]
    use = (asu >= 0) & (gt >= gmin)
    src = np.flatnonzero(use); tgt = asu[use]
    hp = np.zeros(n, bool); hp[tgt] = True
    SU = msu.copy(); SC = msc.copy()
    for k in range(M):
        su, sc = SU[k], SC[k]
        su[src] = tgt; sc[src] = hi
        oth = np.flatnonzero(~use & (su >= 0)); j = su[oth]
        bad = hp[j]
        if drop_contra:
            bad |= (gt[oth] >= gmin) & (root[j] == root[oth]) & (coord[j] != coord[oth] + 1)
        su[oth[bad]] = -1; sc[oth[bad]] = -50.0
    return SU, SC, use

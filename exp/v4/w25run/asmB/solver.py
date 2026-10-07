"""asmB global offset solver: Kruskal-style greedy merging of chain fragments by aggregated offset votes, with lazy
re-aggregation of component-pair votes (heap), hard collision constraints (one node per limb per time coordinate, at most
one of our tiles per coordinate) and a soft completeness check: a coordinate that becomes complete (all 4 limbs) must
hold exactly one of our tiles (0 only for an unanchored tile). LLR per newly complete coordinate: exactly one owned
+log(0.95/0.42), none log(0.05/0.32) (a-priori rates: unanchored share ~5%; random alignment 4*(1/4)*(3/4)^3)."""
import heapq
import numpy as np
from asmlib import fragments

LLR1 = float(np.log(0.95 / 0.42)); LLR0 = float(np.log(0.05 / 0.32))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def tile_frag(d, frag, off):
    n = d["n"]; lim, anch, pos = d["lim"], d["anch"], d["pos"]
    tf = np.full(n, -1, np.int64); to = np.zeros(n, np.int64)
    a = np.flatnonzero(anch); tf[a] = frag[lim[a], pos[a]]; to[a] = off[lim[a], pos[a]]
    return tf, to


def pair_votes(Su, Sc, cand=None, Lo=None, l3w=0.3, l3k=3):
    """aggregated tile-pair votes: (i, j, w) with w = sum_k [Su_k(i)=j] sigmoid(score_k)/members + l3w*sigmoid(lo) (top-l3k L3)"""
    n = Su.shape[1]; M = Su.shape[0]
    I = np.broadcast_to(np.arange(n)[None], Su.shape).ravel(); J = Su.ravel(); P = sig(Sc.ravel()) / M
    ok = (J >= 0); I, J, P = I[ok], J[ok], P[ok]
    if cand is not None and l3w > 0:
        o = np.argsort(-np.where(cand >= 0, Lo, -1e9), 1)[:, :l3k]
        c = np.take_along_axis(cand, o, 1); l = np.take_along_axis(Lo, o, 1)
        I2 = np.broadcast_to(np.arange(n)[:, None], c.shape).ravel(); J2 = c.ravel(); P2 = l3w * sig(l.ravel())
        ok = (J2 >= 0) & (l.ravel() > -49); I = np.r_[I, I2[ok]]; J = np.r_[J, J2[ok]]; P = np.r_[P, P2[ok]]
    ok = I != J; I, J, P = I[ok], J[ok], P[ok]
    key = I * n + J; uk, inv = np.unique(key, return_inverse=True); w = np.bincount(inv, P)
    return uk // n, uk % n, w


def build_votes(d, frag, off, thr, Su, Sc, cand=None, Lo=None, l3w=0.3, l3k=3, chain_w=1.0):
    """votes (fa, fb, dd, w): t(origin fb) = t(origin fa) + dd, weight w (our links / L3 between anchored tiles,
    weak chain links (conf < thr) with weight chain_w * conf)."""
    anch = d["anch"]
    tf, to = tile_frag(d, frag, off)
    i, j, w = pair_votes(Su, Sc, cand, Lo, l3w, l3k)
    ok = anch[i] & anch[j]; i, j, w = i[ok], j[ok], w[ok]
    FA, FB, DD, WW = [tf[i]], [tf[j]], [to[i] + 1 - to[j]], [w]
    for L in range(4):
        su, cf = d["succ"][L], d["conf"][L]
        q = np.flatnonzero((su >= 0) & (cf < thr))
        FA.append(frag[L, q]); FB.append(frag[L, su[q]]); DD.append(off[L, q] + 1 - off[L, su[q]]); WW.append(chain_w * cf[q].astype(np.float64))
    fa, fb, dd, w = (np.concatenate(x) for x in (FA, FB, DD, WW))
    ok = (fa != fb) & (w > 0)
    return fa[ok], fb[ok], dd[ok], w[ok]


class Solver:
    def __init__(self, d, thr=0.9):
        self.d = d; self.thr = thr
        self.frag, self.off, self.flim, self.flen, self.fnodes = fragments(d["succ"], d["conf"], thr)
        self.F = len(self.flim)

    def solve(self, fa, fb, dd, w, wmin=0.8, ratio=2.0, smin=-99.0):
        """smin: reject a merge whose completeness LLR is below smin (-99 = off)"""
        d = self.d; F = self.F; owner = d["owner"]
        parent = np.arange(F); offs = np.zeros(F, np.int64); ver = np.zeros(F, np.int64)
        members = [[f] for f in range(F)]
        cov = []; own = []
        for f in range(F):
            L = int(self.flim[f]); nodes = self.fnodes[f]; ow = owner[L, nodes]
            cov.append({c: [1 << L, int(ow[c] >= 0)] for c in range(len(nodes))})
            own.append({int(c): int(ow[c]) for c in np.flatnonzero(ow >= 0)})
        adj = [[] for _ in range(F)]
        for v in range(len(fa)):
            adj[fa[v]].append(v); adj[fb[v]].append(v)
        self.n_merge = 0; self.n_coll = 0; self.n_srej = 0

        def agg(A, B):
            src = A if len(adj[A]) <= len(adj[B]) else B
            acc = {}
            for v in adj[src]:
                a, b = fa[v], fb[v]
                ra, rb = parent[a], parent[b]
                if ra == A and rb == B:
                    D = offs[a] + dd[v] - offs[b]
                elif ra == B and rb == A:
                    D = offs[b] - offs[a] - dd[v]
                else:
                    continue
                acc[D] = acc.get(D, 0.0) + w[v]
            return acc

        def push(heap, A, B, acc):
            if not acc:
                return
            items = sorted(acc.items(), key=lambda x: -x[1])
            W1 = items[0][1]; W2 = items[1][1] if len(items) > 1 else 0.0
            if W1 >= wmin:
                heapq.heappush(heap, (-W1, int(A), int(B), int(items[0][0]), int(ver[A]), int(ver[B]), W2))

        heap = []; key = {}
        for v in range(len(fa)):
            a, b, D = int(fa[v]), int(fb[v]), int(dd[v])
            if a > b:
                a, b, D = b, a, -D
            k2 = key.setdefault((a, b), {}); k2[D] = k2.get(D, 0.0) + w[v]
        for (a, b), acc in key.items():
            push(heap, a, b, acc)
        del key
        done = set()
        while heap:
            negW, A, B, D, va, vb, W2 = heapq.heappop(heap)
            if parent[A] != A or parent[B] != B or ver[A] != va or ver[B] != vb:
                ra, rb = int(parent[A]), int(parent[B])
                if ra == rb:
                    continue
                if ra > rb:
                    ra, rb = rb, ra
                tag = (ra, rb, int(ver[ra]), int(ver[rb]))
                if tag in done:
                    continue
                done.add(tag)
                push(heap, ra, rb, agg(ra, rb))
                continue
            W1 = -negW
            if W1 < wmin or W1 < ratio * W2:
                continue
            big, small, sh = (A, B, D) if len(members[A]) >= len(members[B]) else (B, A, -D)
            cb, cs = cov[big], cov[small]
            coll = False; z1 = z0 = 0
            for c, (ms, os_) in cs.items():
                e = cb.get(c + sh)
                if e is None:
                    continue
                if (e[0] & ms) or (e[1] + os_ > 1):
                    coll = True; break
                if (e[0] | ms) == 15:
                    if e[1] + os_ == 1:
                        z1 += 1
                    else:
                        z0 += 1
            if coll:
                self.n_coll += 1; continue
            if z1 * LLR1 + z0 * LLR0 < smin:
                self.n_srej += 1; continue
            for f in members[small]:
                offs[f] += sh; parent[f] = big
            members[big].extend(members[small]); members[small] = []
            for c, (ms, os_) in cs.items():
                e = cb.get(c + sh)
                if e is None:
                    cb[c + sh] = [ms, os_]
                else:
                    e[0] |= ms; e[1] += os_
            own[big].update({c + sh: t for c, t in own[small].items()})
            cov[small] = {}; own[small] = {}
            adj[big].extend(adj[small]); adj[small] = []
            ver[big] += 1; ver[small] += 1
            self.n_merge += 1
        self.parent, self.offs, self.members, self.cov, self.own = parent, offs, members, cov, own
        return self

    def groups(self):
        d = self.d; n = d["n"]
        tf, to = tile_frag(d, self.frag, self.off)
        g = np.where(tf >= 0, self.parent[np.maximum(tf, 0)], -1)
        c = np.where(tf >= 0, self.offs[np.maximum(tf, 0)] + to, 0)
        size = np.zeros(n, np.int64); cnt = {}
        for r in g[g >= 0]:
            cnt[r] = cnt.get(r, 0) + 1
        size[g >= 0] = [cnt[r] for r in g[g >= 0]]
        return g, c, size

    def derive(self, gmin=10):
        """assembly successor of every anchored tile in a group of >= gmin tiles: our tile at coordinate c+1, else -1;
        status: 0 not in group, 1 successor found, 2 next coordinate complete but no owned tile, 3 partial, 4 empty"""
        d = self.d; n = d["n"]
        g, c, size = self.groups()
        asu = np.full(n, -1, np.int64); status = np.zeros(n, np.int8)
        for i in np.flatnonzero((g >= 0) & (size >= gmin)):
            r = g[i]; j = self.own[r].get(int(c[i]) + 1, -1)
            if j >= 0:
                asu[i] = j; status[i] = 1
            else:
                e = self.cov[r].get(int(c[i]) + 1)
                status[i] = 4 if e is None else (2 if e[0] == 15 else 3)
        return asu, status, g, c, size

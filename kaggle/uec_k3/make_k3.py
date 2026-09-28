"""Generate kaggle/uec_k3/uec_k3.py: everything of uec_k1.py up to its main() (data/feature/model code, verbatim) plus a
new main that trains the two members 5-fold by subject (the Hanbat notebook's fold split), predicts every 1-s tile of
the held-out subjects' sessions for all four sensors (OOF, our train_meta layout (rows,4,19)) and the test set
(mean over folds)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "uec_k1", "uec_k1.py"), encoding="utf-8").read()
marker = "# ---------------------------------------------------------------- main"
prefix = src[:src.index(marker)]
prefix = prefix.replace('"""wear-uec-k1 : Kaggle GPU script (no internet, no secrets).',
                        '"""wear-uec-k3 : Kaggle GPU script (no internet, no secrets). 5-fold OOF variant of wear-uec-k1.')
prefix = prefix.replace('OUT = r"E:\\Claude code\\wear\\kaggle\\uec_k1\\smoke_out" if LOCAL else "/kaggle/working"',
                        'OUT = r"E:\\Claude code\\wear\\kaggle\\uec_k3\\smoke_out" if LOCAL else "/kaggle/working"')
prefix = prefix.replace("GUARD_START_S = 45 * 60", "GUARD_START_S = int(os.environ.get('WEAR_GUARD_MIN', '54')) * 60")
assert 'uec_k3\\smoke_out' in prefix and "WEAR_GUARD_MIN" in prefix

NEW_MAIN = r'''
# ---------------------------------------------------------------- tiles (every 1-s window, all 4 sensors) for OOF
FOLDS = {0: [5, 14, 17, 21], 1: [0, 4, 9, 15], 2: [3, 7, 11, 12, 19], 3: [1, 8, 10, 16], 4: [2, 6, 13, 18, 20]}
if SMOKE:
    FOLDS = {0: [5], 1: [9], 2: [20]}


def build_tiles(sess):
    lab, A, sid = read_session(os.path.join(ROOT, "train", "inertial_feat", f"{sess}.csv"))
    T = len(lab)
    V, src = load_frames(sess, T)
    ts = np.arange(T // WS); starts = ts * WS
    centre = np.rint((starts + WS / 2.0) * VIDEO_HZ / IMU_HZ).astype(np.int64); lo = centre - VWIN // 2
    ok = (lo >= 0) & (lo + VWIN <= V.shape[0])
    ts, starts, lo = ts[ok], starts[ok], lo[ok]
    n = len(ts)
    fidx = (lo[:, None] + np.arange(VWIN)[None, :]).ravel()
    frames = np.asarray(V[fidx], np.float32).reshape(n, VWIN, VIDEO_DIM)
    vvalid = np.isfinite(frames).all(axis=(1, 2))
    vm, vs, vd, sc = video_feats(frames); del frames
    W = A[(starts[:, None] + np.arange(WS)[None, :]).ravel()].reshape(n, WS, 4, 3).transpose(0, 2, 1, 3)   # (n,4,50,3)
    valid = np.isfinite(W).all(axis=(2, 3))                                                                  # (n,4)
    X = np.stack([make_cnn8(W[:, k]) for k in range(4)], 1)                                                  # (n,4,8,50)
    return dict(session=sess, sbj=sid, t=ts, X=X, vm=vm, vs=vs, vd=vd, sc=sc, valid=valid, vvalid=vvalid)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    log(f"device={dev} LOCAL={LOCAL} SMOKE={SMOKE} epochs={EPOCHS} max_batches={MAX_BATCHES} models={MODELS} folds={FOLDS}")
    all_sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "train", "inertial_feat", "sbj_*.csv")))
    train_sessions = [s for s in all_sessions if not (len(s.split("_")) >= 3 and s.split("_")[-1] == "2")]
    if SMOKE:
        all_sessions = [s for s in SMOKE_SESSIONS if s in all_sessions]; train_sessions = [s for s in train_sessions if s in all_sessions]
    log(f"sessions all={len(all_sessions)} train={len(train_sessions)}")

    parts, stats, psbj = [], [], []
    for s in train_sessions:
        X, y, vm, vs, vd, sc, st = build_session(s)
        parts.append((X, y, vm, vs, vd, sc)); stats.append(st); psbj.append(np.full(len(y), st["sbj"], np.int64)); log(st)
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    VM = np.concatenate([p[2] for p in parts]); VS = np.concatenate([p[3] for p in parts])
    VD = np.concatenate([p[4] for p in parts]); SC = np.concatenate([p[5] for p in parts]); WSBJ = np.concatenate(psbj)
    del parts
    Nw = len(y)
    Xs = np.ascontiguousarray(X.transpose(1, 0, 2, 3)).reshape(4 * Nw, 8, WS); del X
    tile = np.tile(np.arange(Nw), 4)
    sens = np.repeat(np.array([SENSOR_TO_ID[k] for k in SENSOR_KEYS], np.int64), Nw)
    ys = np.tile(y, 4); ssbj = np.tile(WSBJ, 4)
    log(f"windows={Nw} samples={len(ys)} subjects={sorted(set(WSBJ.tolist()))}")

    tiles = [build_tiles(s) for s in all_sessions]
    rows = pd.DataFrame({"session": np.concatenate([[d["session"]] * len(d["t"]) for d in tiles]),
                         "sbj": np.concatenate([[d["sbj"]] * len(d["t"]) for d in tiles]),
                         "t": np.concatenate([d["t"] for d in tiles])})
    rows.to_csv(os.path.join(OUT, "oof_rows.csv"), index=False)
    NR = len(rows); off = np.r_[0, np.cumsum([len(d["t"]) for d in tiles])]
    log(f"tile rows={NR} (expected 69326 on full data)")
    tile_valid = np.concatenate([d["valid"] for d in tiles]); np.save(os.path.join(OUT, "oof_valid.npy"), tile_valid)

    Xte, sens_te, tVM, tVS, tVD, tSC, NT = build_test()
    log(f"test rows={NT}")
    OOF = {m: np.full((NR, 4, NC), np.nan, np.float32) for m in MODELS}
    TEST = {m: [] for m in MODELS}
    summary = dict(sessions=stats, windows=int(Nw), tile_rows=int(NR), epochs=EPOCHS, folds={k: v for k, v in FOLDS.items()}, runs=[])

    def save_all():
        for m in MODELS:
            np.save(os.path.join(OUT, f"oof_{m}.npy"), OOF[m])
            if TEST[m]:
                np.save(os.path.join(OUT, f"test_{m}_folds.npy"), np.stack(TEST[m]))
                np.save(os.path.join(OUT, CFG[m]["out"]), np.mean(TEST[m], 0).astype(np.float32))
        json.dump(summary, open(os.path.join(OUT, "k3_summary.json"), "w"), indent=1)

    @torch.inference_mode()
    def predict(model, feats, norm):
        model.eval(); outs = []
        cm, cs, sm, ss = norm
        Xf, vm, vs, vd, sc, sn = feats
        for b in range(0, len(sn), 4096):
            s = slice(b, b + 4096)
            lo = model(torch.from_numpy((Xf[s] - cm) / cs).to(dev), torch.from_numpy(vm[s]).to(dev), torch.from_numpy(vs[s]).to(dev),
                       torch.from_numpy(vd[s]).to(dev), torch.from_numpy((sc[s] - sm) / ss).to(dev), torch.from_numpy(sn[s]).to(dev))
            outs.append(torch.softmax(lo.float(), 1).cpu())
        return torch.cat(outs).numpy().astype(np.float32)

    for f, held in FOLDS.items():
        if time.time() - T0 > GUARD_START_S:
            log(f"fold {f}: time guard, skipped"); break
        tr = ~np.isin(ssbj, held); n = int(tr.sum())
        cnn_mean = Xs[tr].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        cnn_std = np.maximum(Xs[tr].std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None], 1e-6)
        tw = tile[tr]; sc_mean = SC[tw].mean(0, dtype=np.float64).astype(np.float32); sc_std = np.maximum(SC[tw].std(0, dtype=np.float64).astype(np.float32), 1e-6)
        norm = (cnn_mean, cnn_std, sc_mean, sc_std)
        def to_dev(a, dt):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dt).to(dev)
        train_t = dict(X=to_dev((Xs[tr] - cnn_mean) / cnn_std, torch.float32), VM=to_dev(VM, torch.float32), VS=to_dev(VS, torch.float32),
                       VD=to_dev(VD, torch.float32), SC=to_dev((SC - sc_mean) / sc_std, torch.float32),
                       tile=to_dev(tile[tr], torch.long), sens=to_dev(sens[tr], torch.long), y=to_dev(ys[tr], torch.long))
        held_sessions = [i for i, d in enumerate(tiles) if d["sbj"] in held]
        log(f"== fold {f} held={held} train samples={n} held sessions={[tiles[i]['session'] for i in held_sessions]}")
        for name in MODELS:
            if time.time() - T0 > GUARD_START_S:
                log(f"fold {f} {name}: time guard, skipped"); break
            cfg = CFG[name]
            torch.manual_seed(SEED + f); np.random.seed(SEED + f)
            model = CNN8VideoAuxNet(cfg).to(dev)
            counts = np.maximum(np.bincount(ys[tr], minlength=NC).astype(np.float32), 1.0)
            w = np.sqrt(n / (NC * counts)); w = w / w.mean()
            crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=cfg["label_smoothing"])
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            g = torch.Generator().manual_seed(SEED + f); lr = cfg["learning_rate"]; hist = []
            for ep in range(1, EPOCHS[name] + 1):
                if time.time() - T0 > GUARD_START_S:
                    log(f"fold {f} {name}: time guard, not starting epoch {ep}"); break
                if ep in LR_HALVE_BEFORE:
                    lr *= 0.5
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                model.train(); te = time.time(); tl = 0.0; nb = 0
                order = torch.randperm(n, generator=g).to(dev)
                for b in range(0, n, BATCH):
                    if MAX_BATCHES is not None and nb >= MAX_BATCHES:
                        break
                    idx = order[b:b + BATCH]
                    if len(idx) < 2:
                        continue
                    tt = train_t["tile"][idx]
                    logits = model(train_t["X"][idx], train_t["VM"][tt], train_t["VS"][tt], train_t["VD"][tt], train_t["SC"][tt], train_t["sens"][idx])
                    loss = crit(logits, train_t["y"][idx])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    tl += float(loss.item()); nb += 1
                hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), lr=lr, batches=nb, epoch_sec=round(time.time() - te, 1)))
                log(f"fold {f} {name} epoch={ep} loss={tl / max(nb, 1):.4f} epoch_sec={time.time() - te:.0f} elapsed={time.time() - T0:.0f}s")
            # OOF: every tile of the held-out subjects' sessions, all 4 sensors
            for i in held_sessions:
                d = tiles[i]; nt = len(d["t"])
                for k, key in enumerate(SENSOR_KEYS):
                    P = predict(model, (d["X"][:, k], d["vm"], d["vs"], d["vd"], d["sc"], np.full(nt, SENSOR_TO_ID[key], np.int64)), norm)
                    P[~d["valid"][:, k]] = np.nan
                    OOF[name][off[i]:off[i + 1], k] = P
            Pt = predict(model, (Xte, tVM, tVS, tVD, tSC, sens_te), norm); TEST[name].append(Pt)
            acc = [np.nanmean(np.nanmean(OOF[name][off[i]:off[i + 1]], 1).argmax(1) == 0) for i in held_sessions[:1]]
            summary["runs"].append(dict(fold=f, model=name, epochs=len(hist), history=hist, elapsed=round(time.time() - T0)))
            log(f"fold {f} {name}: OOF done, test pred_dist={np.bincount(Pt.argmax(1), minlength=NC).tolist()}")
            del model, opt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del train_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        save_all()
    save_all()
    log(f"DONE elapsed {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
'''
out = prefix + NEW_MAIN
open(os.path.join(HERE, "uec_k3.py"), "w", encoding="utf-8").write(out)
import json
json.dump({"id": "koushikrudra/wear-uec-k3", "title": "wear-uec-k3", "code_file": "uec_k3.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
           "dataset_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"], "kernel_sources": [],
           "model_sources": []}, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=1)
print("wrote uec_k3.py", len(out.splitlines()), "lines")

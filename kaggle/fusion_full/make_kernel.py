"""Generate fusion_full.py = fusion.py data/model code + full-data multi-seed training with a hard time guard."""
src = open(r"E:\Claude code\wear\kaggle\fusion\fusion.py", encoding="utf-8").read()
head = src[: src.index("subjects = np.unique(sbj); rng")]
head = head.replace('"""Multimodal fusion net', '"""FULL-DATA multi-seed refit of the multimodal fusion net', 1)
tail = r'''
# ------------------------------------------------------------------ full-data multi-seed refit (GPU budget guard)
SEEDS = [0, 1, 2]; BUDGET_START_S = 40 * 60          # never START a new seed after 40 min of wall time
rng = np.random.RandomState(0)
cnt = np.bincount(y_sec[rows_sec], minlength=NC); keep_null = int(3 * np.median(cnt[1:]))
te_secs = np.arange(len(tm)); preds = []
for sd in SEEDS:
    if time.time() - t0 > BUDGET_START_S:
        log(f"time guard: skip seed {sd}"); break
    torch.manual_seed(sd); np.random.seed(sd); r = np.random.RandomState(sd)
    trs, trl = rows_sec.copy(), rows_limb.copy()
    nidx = np.where(y_sec[trs] == 0)[0]; sel = np.ones(len(trs), bool)
    if len(nidx) > keep_null: sel[r.choice(nidx, len(nidx) - keep_null, replace=False)] = False
    trs, trl = trs[sel], trl[sel]
    cnt2 = np.bincount(y_sec[trs], minlength=NC).astype(np.float64)
    w_cls = torch.tensor((cnt2.mean() / np.maximum(cnt2, 1)) ** 0.5, dtype=torch.float32, device=dev)
    dl = torch.utils.data.DataLoader(DS(trs, trl, True), batch_size=512, shuffle=True, num_workers=0 if LOCAL else 3, drop_last=True)
    model = Net(DV).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = EPOCHS * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    log(f"seed {sd}: train rows {len(trs)} (null kept {keep_null}), steps/epoch {len(dl)}")
    for ep in range(EPOCHS):
        model.train(); tl = 0; nb = 0
        for x, v, l, yy in dl:
            x, v, l, yy = x.to(dev), v.to(dev), l.to(dev), yy.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                lo, li, lv = model(x, v, l)
                loss = Fnn.cross_entropy(lo, yy, weight=w_cls, label_smoothing=0.05) + 0.3 * (Fnn.cross_entropy(li, yy, weight=w_cls) + Fnn.cross_entropy(lv, yy, weight=w_cls))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        log(f"seed {sd} ep {ep}: loss {tl/nb:.4f}")
    p = predict(model, te_secs, limb_te, imu_src=XI_te, vid_src=VID_te).astype(np.float32)
    np.save(f"{OUT}/test_full_s{sd}.npy", p); preds.append(p)
    np.save(f"{OUT}/test.npy", np.mean(preds, 0).astype(np.float32))          # saved after every seed
    log(f"seed {sd} done; saved test.npy from {len(preds)} seed(s)")
log("DONE", len(preds), "seeds")
'''
open(r"E:\Claude code\wear\kaggle\fusion_full\fusion_full.py", "w", encoding="utf-8").write(head + tail)
print("written", len(head + tail), "chars")

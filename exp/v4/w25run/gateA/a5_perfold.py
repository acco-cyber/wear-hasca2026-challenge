"""Gate A: per-fold kernel baselines (K7, K9 own links) and paired per-fold deltas of the gate-A decodes."""
import os, sys, numpy as np
W = r"E:\Claude code\wear"
sys.path.insert(0, os.path.join(W, "exp", "hyb"))
from hanbat_stack import macro_f1
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4"); K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
G = os.path.join(W, "exp", "v4", "w25run", "gateA")
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64)
pf = lambda lab: [round(macro_f1(y[fold == f], lab[fold == f]), 4) for f in range(5)]
rows = {}
for nm, d in (("K7", K7), ("K9", K9)):
    s = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
    rows[nm + "_kernel_pre"] = s["QB_OOF"].argmax(1); rows[nm + "_kernel_ref"] = s["ref_oof"].astype(np.int64)
rows["K7_chain_ref"] = np.load(os.path.join(W, "subs", "sub_v4l_w25simA_k7_labo.npy")).astype(np.int64)
rows["K7_chain_pre"] = np.load(os.path.join(W, "subs", "sub_v4l_w25simA_k7_Qo.npy")).argmax(1)
rows["K9_chain_ref"] = np.load(os.path.join(W, "subs", "sub_v4l_w25simA_k9_labo.npy")).astype(np.int64)
rows["K9_chain_pre"] = np.load(os.path.join(W, "subs", "sub_v4l_w25simA_k9_Qo.npy")).argmax(1)
rows["fused_base_pre"] = np.load(os.path.join(G, "subs", "sub_v4c_w25simA_base_kernel_Qo.npy")).argmax(1)
rows["fused_base_ref"] = np.load(os.path.join(G, "subs", "sub_v4c_w25simA_base_kernel_labo.npy")).astype(np.int64)
rows["fused_chain_pre"] = np.load(os.path.join(W, "subs", "sub_v4c_w25simA_fused_Qo.npy")).argmax(1)
rows["fused_chain_ref_ownrefinerlinks"] = np.load(os.path.join(W, "subs", "sub_v4c_w25simA_fused_labo.npy")).astype(np.int64)
rows["fused_chain_ref_chainrefinerlinks"] = np.load(os.path.join(G, "subs", "sub_v4c_w25simA_fused_chref_labo.npy")).astype(np.int64)
for k, lab in rows.items():
    print(f"{k:>36}: OOF {macro_f1(y, lab):.4f} per fold {pf(lab)}")
for a_, b_ in (("K7_chain_ref", "K7_kernel_ref"), ("K9_chain_ref", "K9_kernel_ref"), ("fused_chain_pre", "fused_base_pre"),
               ("fused_chain_ref_ownrefinerlinks", "fused_base_ref"), ("fused_chain_ref_chainrefinerlinks", "fused_base_ref")):
    d = np.array(pf(rows[a_])) - np.array(pf(rows[b_]))
    print(f"delta {a_} - {b_}: {macro_f1(y, rows[a_]) - macro_f1(y, rows[b_]):+.4f} | per fold {np.round(d, 4).tolist()} ({(d > 0).sum()}/5 up)")

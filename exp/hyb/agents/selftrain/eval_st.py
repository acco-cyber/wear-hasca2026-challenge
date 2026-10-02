"""Assemble self-trained S3/T OOF from runs/<tag> (missing folds -> baseline), rebuild the tab blend and score:
tab argmax, full recipe (L0 + our chain links, prior .3 counts .3 gate .55:top2) and the plain hanbat graph (L0), per fold.
python eval_st.py <tag> [<tag2> ...] [--mix S3,T] [--plain]"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def assemble(tag, v, base, fold):
    out = base.copy(); got = []
    for f in range(5):
        p = os.path.join(HERE, "runs", tag, f"{v}_f{f}.npy")
        if os.path.exists(p):
            out[fold == f] = np.load(p); got.append(f)
    return out, got


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--plain", action="store_true")
    ap.add_argument("--only", default="both", help="both|S3|T : which expert to take from the run (others baseline)")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    sm = load_meta(); y, fold = sm["y"], sm["fold"]
    win, S3b, Tb, LBb = base_parts(); ours, has = ours_oof(len(y))
    res_path = os.path.join(HERE, "results.tsv")
    for tag in a.tags:
        S3, g3 = assemble(tag, "S3", S3b, fold) if a.only in ("both", "S3") else (S3b, [])
        T, gT = assemble(tag, "T", Tb, fold) if a.only in ("both", "T") else (Tb, [])
        LB = tab_blend(win, S3, T)
        name = tag if a.only == "both" else f"{tag}[{a.only}]"
        line = [f"{name}: S3 folds {g3} T folds {gT}",
                f"  S3 F1 {macro_f1(y, S3.argmax(1)):.4f} (base {macro_f1(y, S3b.argmax(1)):.4f})  T F1 {macro_f1(y, T.argmax(1)):.4f} (base {macro_f1(y, Tb.argmax(1)):.4f})",
                f"  tab F1 {macro_f1(y, LB.argmax(1)):.4f} (base {macro_f1(y, LBb.argmax(1)):.4f}) per fold " +
                " ".join(f"{u:.4f}" for u in per_fold(y, LB.argmax(1), fold))]
        lab, *_ = recipe(graph_dd_oof(LB, sm, extra=True), ours, has)
        pf = per_fold(y, lab, fold)
        line.append(f"  RECIPE F1 {macro_f1(y, lab):.4f} per fold " + " ".join(f"{u:.4f}" for u in pf))
        if a.plain:
            labp = plain_graph(LB, sm); line.append(f"  plain graph F1 {macro_f1(y, labp):.4f} per fold " + " ".join(f"{u:.4f}" for u in per_fold(y, labp, fold)))
        print("\n".join(line), flush=True)
        with open(res_path, "a") as fh:
            fh.write("\t".join([name, f"{macro_f1(y, LB.argmax(1)):.4f}", f"{macro_f1(y, lab):.4f}"] + [f"{u:.4f}" for u in pf]) + "\n")
        if a.save:
            np.save(os.path.join(HERE, "runs", tag, "oof_logp_b.npy"), LB); np.save(os.path.join(HERE, "runs", tag, "recipe_lab.npy"), lab)


if __name__ == "__main__":
    main()

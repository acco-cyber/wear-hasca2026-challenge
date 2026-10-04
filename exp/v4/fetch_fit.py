"""Download the decode inputs a v4 fork kept (keep4/, without the bulky per-run raw outputs) plus its submission,
final probabilities and log.   KAGGLE_API_TOKEN must be set in the environment.
  python fetch_fit.py <owner/kernel> [<out_dir>]      (default out_dir: work/v4/<kernel-name>)"""
import os, sys, subprocess
W = r"E:\Claude code\wear"
FILES = ["keep4/stage.npz", "keep4/links.npz", "keep4/link_logodds.npz", "keep4/dec_cache.npz", "keep4/oof_emb.npy",
         "keep4/test_emb.npy", "keep4/tile_scalars.npz", "submission.csv", "final_probabilities.npz", "log"]


def main():
    ref = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(W, "work", "v4", ref.split("/")[1])
    os.makedirs(out, exist_ok=True)
    tok = os.environ["KAGGLE_API_TOKEN"]
    rc = subprocess.run([sys.executable, os.path.join(W, "fetch_output.py"), tok, ref, out] + FILES).returncode
    have = [f for f in FILES[:-1] if os.path.exists(os.path.join(out, f))]
    print(f"{ref}: rc {rc}, have {len(have)}/{len(FILES) - 1}: missing {[f for f in FILES[:-1] if f not in have]}")


if __name__ == "__main__":
    main()

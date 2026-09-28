"""Sequential driver for dec_hanbat.py runs (keeps CPU contention bounded). python run_dec.py <set>"""
import os, sys, subprocess
W = r"E:\Claude code\wear"; py = sys.executable; D = os.path.join(W, "exp", "hyb", "dec_hanbat.py")
E39 = "uec_ens:1.0,v3b_full:0.4,aka_full:0.3"
SETS = {
    "A": [("hb1_tab", "hb_tab:1.0", []), ("hb2_e39_tab", f"{E39},hb_tab:1.0", ["--votes"]), ("hb3_graph", "hb_graph:1.0", [])],
    "B": [("hb4_e39_tab_L2", f"{E39},hb_tab:1.0", ["--votes", "--struct", "hbL2"]), ("hb5_tab_L2", "hb_tab:1.0", ["--struct", "hbL2"])],
}
for name, extra, flags in SETS[sys.argv[1]]:
    print("=== run", name, extra, flags, flush=True)
    subprocess.run([py, D, "run", name, extra, *flags], check=False, env=dict(os.environ, PYTHONPATH=os.path.join(W, "shim")))

"""Wrapper around the lead's exp/decoder/sim_oof.py (variant mrf4, eval+extra sessions) with short model names.
python run_simoof.py "name=m1:w1+m2:w2;name2=..."   (m: exp/base dir name or w:<work dir>)"""
import os, sys, subprocess
EXP = r"E:\Claude code\wear\exp\base"; WORK = r"E:\Claude code\wear\work"
def path(m): return os.path.join(WORK, m[2:], "oof.npy") if m.startswith("w:") else os.path.join(EXP, m, "oof.npy")
items = []
for item in sys.argv[1].split(";"):
    name, rest = item.split("=")
    parts = []
    for p in rest.split("+"):
        if p.count(":") > (1 if p.startswith("w:") else 0): m, w = p.rsplit(":", 1)
        else: m, w = p, "1"
        parts.append(f"{path(m)}:{w}")
    items.append(f"{name}=" + ",".join(parts))
variant = sys.argv[2] if len(sys.argv) > 2 else "mrf4"
env = dict(os.environ); env["OMP_NUM_THREADS"] = "4"
r = subprocess.run([sys.executable, "-u", r"E:\Claude code\wear\exp\decoder\sim_oof.py", "--variant", variant, "--oofs", ";".join(items)],
                   cwd=r"E:\Claude code\wear\exp\decoder", env=env, capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
with open(os.path.join(EXP, "simoof_log.txt"), "a") as f: f.write(r.stdout)

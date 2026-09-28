"""Run the 5 std folds of train_uec.py (tile rows, fixed iterations) with up to 3 concurrent processes x 4 threads.
python run_std.py [--iters 600] [--lr 0.08] [--par 3]"""
import os, sys, time, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=600); ap.add_argument("--lr", type=float, default=0.08)
ap.add_argument("--par", type=int, default=3); ap.add_argument("--folds", default="0,1,2,3,4")
a = ap.parse_args()
env = dict(os.environ, THREADS="4", OMP_NUM_THREADS="4", PYTHONPATH=r"E:\Claude code\wear\shim")
todo = [int(f) for f in a.folds.split(",")]; running = {}
while todo or running:
    while todo and len(running) < a.par:
        f = todo.pop(0)
        log = open(os.path.join(HERE, f"std_f{f}.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "train_uec.py"), "std", str(f), "--tile_only", "--no_es",
                              "--iters", str(a.iters), "--lr", str(a.lr)], stdout=log, stderr=subprocess.STDOUT, env=env)
        running[f] = (p, log); print(time.strftime("%H:%M:%S"), "started fold", f, flush=True)
    time.sleep(20)
    for f, (p, log) in list(running.items()):
        if p.poll() is not None:
            log.close(); print(time.strftime("%H:%M:%S"), "fold", f, "exit", p.returncode, flush=True); del running[f]
print("all done")

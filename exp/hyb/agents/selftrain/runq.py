"""Run st_cv.py configurations sequentially: python runq.py <queue_name> [--after PID] -- "<args1>" "<args2>" ...
each config's stdout/stderr -> logs/<tag>.log/.err"""
import os, sys, subprocess, time, shlex
HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; name = argv.pop(0); after = None
if argv and argv[0] == "--after":
    after = int(argv[1]); argv = argv[2:]
if argv and argv[0] == "--":
    argv = argv[1:]


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


if after:
    while alive(after):
        time.sleep(20)
for cfg in argv:
    parts = shlex.split(cfg); script = "st_cv.py"
    if parts[0].endswith(".py"):
        script = parts.pop(0)
    tag = parts[parts.index("--tag") + 1]
    with open(os.path.join(HERE, "logs", f"{tag}.log"), "a") as fo, open(os.path.join(HERE, "logs", f"{tag}.err"), "a") as fe:
        print(time.strftime("%H:%M:%S"), name, "start", script, cfg, flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)] + parts, stdout=fo, stderr=fe, cwd=HERE)
        print(time.strftime("%H:%M:%S"), name, "done", cfg, "rc", r.returncode, flush=True)

"""Dump code+outputs of public kernels that mention train shapes / label stats (fallback for task 5 while train CSVs download)."""
import json, sys, glob, os, re
root = r"E:\Claude code\wear\kernels"
keys = sys.argv[1:] or ["shape", "null", "value_counts", "fps", "frames", "duration", "bout"]
for p in sorted(glob.glob(os.path.join(root, "*", "*.ipynb"))):
    nb = json.load(open(p, encoding="utf-8"))
    print("=" * 100); print(p)
    for ci, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        outs = []
        for o in c.get("outputs", []) if c["cell_type"] == "code" else []:
            if "text" in o: outs.append("".join(o["text"]))
            elif "data" in o and "text/plain" in o["data"]: outs.append("".join(o["data"]["text/plain"]))
        txt = "\n".join(outs)
        if txt.strip() and any(k in (src + txt).lower() for k in keys):
            print(f"--- cell {ci} SRC:\n{src[:1200]}\n--- OUT:\n{txt[:2500]}")

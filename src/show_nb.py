"""Print a notebook's cells compactly. python show_nb.py <nb.ipynb> [max_code_chars]"""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8")); mx = int(sys.argv[2]) if len(sys.argv) > 2 else 700
print(len(nb["cells"]), "cells")
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if c["cell_type"] == "markdown":
        print(f"--- MD {i}:", s[:1500])
    else:
        print(f"--- CODE {i} ({len(s)} chars):", s[:mx].replace("\n", " | "))
        for o in c.get("outputs", [])[:3]:
            t = "".join(o.get("text", [])) if "text" in o else str(o.get("data", {}).get("text/plain", ""))[:300]
            if t: print("    OUT:", t[:600].replace("\n", " | "))

import json, sys
p = sys.argv[1]
nb = json.load(open(p, encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if any(k in s for k in ["stride", "window_size", "sampling", "fps", "label_map", "LABEL", "labels"]):
            print("-----"); print(s[:3000])

"""Assemble wear_hanbat_l2oof.py from template.py + verbatim line blocks of woominyo's notebook code (Apache 2.0)."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"E:\Claude code\wear\public_src\woominyo\nb_code.py"
nb = open(SRC, encoding="utf-8").read().splitlines()
tpl = open(os.path.join(HERE, "template.py"), encoding="utf-8").read()


def block(m):
    a, b = int(m.group(1)), int(m.group(2))
    lines = nb[a - 1:b]
    assert not any(l.startswith("%%") for l in lines), (a, b)
    return f"# ---- verbatim: notebook lines {a}-{b}\n" + "\n".join(lines)


out = re.sub(r"#@@BLOCK:(\d+)-(\d+)@@", block, tpl)
dst = os.path.join(HERE, "wear_hanbat_l2oof.py")
open(dst, "w", encoding="utf-8", newline="\n").write(out)
compile(out, dst, "exec")
meta = {"id": "koushikrudra/wear-hanbat-l2oof", "title": "wear-hanbat-l2oof", "code_file": "wear_hanbat_l2oof.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"],
        "kernel_sources": ["koushikrudra/wear-hanbat-gpu"], "model_sources": []}
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=1)
print("wrote", dst, len(out.splitlines()), "lines")

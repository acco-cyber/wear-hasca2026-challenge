import json, sys, glob, os

def dump(path, out):
    nb = json.load(open(path, encoding='utf-8'))
    lines = []
    for i, c in enumerate(nb.get('cells', [])):
        src = ''.join(c.get('source', []))
        lines.append(f'##### CELL {i} [{c.get("cell_type")}]')
        lines.append(src)
        for o in c.get('outputs', []) or []:
            t = o.get('text')
            if t is None and 'data' in o:
                t = o['data'].get('text/plain')
            if t:
                t = ''.join(t) if isinstance(t, list) else t
                if len(t) > 3000:
                    t = t[:1500] + '\n...[cut]...\n' + t[-1500:]
                lines.append('----- OUTPUT')
                lines.append(t)
    open(out, 'w', encoding='utf-8').write('\n'.join(lines))
    return len(nb.get('cells', []))

root = sys.argv[1]
for p in glob.glob(os.path.join(root, '**', '*.ipynb'), recursive=True):
    out = p[:-6] + '.txt'
    n = dump(p, out)
    print(n, os.path.getsize(out), out)

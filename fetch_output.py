"""Download specific output files of a Kaggle kernel with resume support.
python fetch_output.py <token> <kernel_ref> <out_dir> <fileName> [fileName ...]"""
import os, sys, time, urllib.request
os.environ["KAGGLE_API_TOKEN"] = sys.argv[1]
from kaggle.api.kaggle_api_extended import KaggleApi
ref, out, wanted = sys.argv[2], sys.argv[3], sys.argv[4:]
api = KaggleApi(); api.authenticate()
user, kern = ref.split("/")
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
files = []
os.makedirs(out, exist_ok=True)
with api.build_kaggle_client() as kaggle:
    token = None
    while True:
        req = ApiListKernelSessionOutputRequest(); req.user_name = user; req.kernel_slug = kern; req.page_size = 50
        if token: req.page_token = token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        files += list(resp.files or [])
        if "log" in wanted and resp.log and not token:
            open(os.path.join(out, kern + ".log"), "w", encoding="utf-8").write(resp.log); print("log saved")
        token = resp.next_page_token
        if not token: break
names = [f.file_name for f in files]
print("files:", [n for n in names if "/" not in n or n.count("/") <= 1 and not n.startswith(("src/", "cache/"))] if len(names) > 60 else names)
os.makedirs(out, exist_ok=True)
for f in files:
    name = f.file_name; url = f.url
    if name not in wanted: continue
    dest = os.path.join(out, name); tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    t0 = time.time()
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as fh:
                total = int(r.headers.get("Content-Length", 0)) + have
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    fh.write(b); have += len(b)
                    if have % (20 << 20) < (1 << 20): print(f"  {name}: {have/1e6:.0f}/{total/1e6:.0f} MB  {have/1e6/(time.time()-t0+1e-9):.2f} MB/s", flush=True)
            if total and have < total: raise IOError(f"short read {have}/{total}")
            break
        except Exception as e:
            print("  retry", attempt, e, flush=True); time.sleep(3)
    os.replace(tmp, dest); print("done", name, os.path.getsize(dest), flush=True)

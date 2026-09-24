"""Push a set of local files to a GitHub repo as ONE commit using the Git Data API (no git binary needed).
Token is read from the GH_TOKEN environment variable and never written to disk.
python gh_push.py <owner/repo> <branch> "<commit message>" <manifest.txt>
manifest lines:  <local path> => <repo path>      (or  DELETE => <repo path>)
"""
import os, sys, json, base64, urllib.request

API = "https://api.github.com"
tok = os.environ["GH_TOKEN"]

def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json",
                                          "User-Agent": "wear-push", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def main():
    repo, branch, msg, manifest = sys.argv[1:5]
    entries = []
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        src, dst = [x.strip() for x in line.split("=>")]
        entries.append((src, dst.replace("\\", "/")))
    ref = call("GET", f"/repos/{repo}/git/ref/heads/{branch}")
    head = ref["object"]["sha"]
    base_tree = call("GET", f"/repos/{repo}/git/commits/{head}")["tree"]["sha"]
    tree = []
    for src, dst in entries:
        if src == "DELETE":
            tree.append({"path": dst, "mode": "100644", "type": "blob", "sha": None}); continue
        data = open(src, "rb").read()
        try:
            content = data.decode("utf-8"); blob = call("POST", f"/repos/{repo}/git/blobs", {"content": content, "encoding": "utf-8"})
        except UnicodeDecodeError:
            blob = call("POST", f"/repos/{repo}/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        tree.append({"path": dst, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print("blob", dst, len(data), flush=True)
    new_tree = call("POST", f"/repos/{repo}/git/trees", {"base_tree": base_tree, "tree": tree})
    commit = call("POST", f"/repos/{repo}/git/commits", {"message": msg, "tree": new_tree["sha"], "parents": [head]})
    call("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", {"sha": commit["sha"], "force": False})
    print("pushed", commit["sha"], "->", f"https://github.com/{repo}/commit/{commit['sha']}")

if __name__ == "__main__":
    main()

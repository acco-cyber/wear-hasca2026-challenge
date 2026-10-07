"""Anonymous read of Kaggle's public web JSON for the competition (looks for leaderboard split %).
No credentials are used or printed."""
import json, os, re, sys
import requests

OUT = os.path.dirname(os.path.abspath(__file__))
s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"
r = s.get("https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026/leaderboard", timeout=60)
print("page", r.status_code, len(r.text))
xsrf = s.cookies.get("XSRF-TOKEN")
print("xsrf cookie present:", bool(xsrf))
hdr = {"content-type": "application/json"}
if xsrf:
    hdr["x-xsrf-token"] = xsrf
for slug in ["3rd-wear-dataset-challenge-hasca-2026", "2nd-wear-dataset-challenge"]:
    for svc, body in [
        ("competitions.CompetitionService/GetCompetition", {"competitionName": slug}),
    ]:
        try:
            rr = s.post("https://www.kaggle.com/api/i/" + svc, data=json.dumps(body), headers=hdr, timeout=60)
            print(slug, svc, rr.status_code, len(rr.text))
            if rr.status_code == 200:
                fn = os.path.join(OUT, f"web_{slug}_{svc.split('/')[-1]}.json")
                open(fn, "w", encoding="utf-8").write(rr.text)
                d = rr.json()
                for k, v in (d.items() if isinstance(d, dict) else []):
                    if re.search(r"(?i)(public|private|percent|leaderboard|split|external|data)", k):
                        print("  ", k, "=", str(v)[:200])
        except Exception as e:
            print(slug, svc, "ERR", repr(e)[:200])

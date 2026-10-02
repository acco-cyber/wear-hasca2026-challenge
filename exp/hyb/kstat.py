"""Board, latest submissions, and whether the account has entered the 2nd WEAR challenge (read-only)."""
import os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
c = "3rd-wear-dataset-challenge-hasca-2026"
try:
    lb = api.competition_leaderboard_view(c)
    for i, r in enumerate(lb[:10]):
        print(i + 1, r.team_name, r.score)
except Exception as e:
    print("lb err", str(e)[:200])
try:
    for s in api.competition_submissions(c)[:6]:
        print(s.date, s.file_name, s.public_score, s.status)
except Exception as e:
    print("subs err", str(e)[:200])
try:
    r = api.competitions_list(search="wear dataset challenge"); comps = getattr(r, "competitions", r)
    for x in comps:
        if "wear-dataset" in x.ref:
            print(x.ref, "entered:", getattr(x, "user_has_entered", None), "deadline", getattr(x, "deadline", None))
except Exception as e:
    print("list err", str(e)[:200])
if "--dl" in sys.argv:
    out = r"E:\Claude code\wear\data\wear2025"
    try:
        api.competition_download_file("2nd-wear-dataset-challenge", "test.csv", path=out, quiet=True); print("downloaded", os.listdir(out))
    except Exception as e:
        print("download FAIL", str(e)[:160])

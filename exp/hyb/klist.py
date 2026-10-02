"""List recent public kernels and datasets for the competition, plus the board (read-only)."""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
c = "3rd-wear-dataset-challenge-hasca-2026"
for sort in ("dateRun", "voteCount"):
    print("== kernels by", sort)
    try:
        for k in api.kernels_list(competition=c, sort_by=sort, page_size=20):
            print(k.last_run_time, "|", k.ref, "|", k.title, "| votes", k.total_votes)
    except Exception as e:
        print("err", str(e)[:200])
print("== datasets")
seen = set()
for q in ("wear hasca", "wear timeline", "wear 2026", "wear challenge", "wear-hasca"):
    try:
        for d in api.dataset_list(search=q, sort_by="updated"):
            if d.ref not in seen and ("wear" in d.ref.lower() or "wear" in (d.title or "").lower() or "hasca" in (d.title or "").lower()):
                seen.add(d.ref); print(d.last_updated, "|", d.ref, "|", d.title, "|", getattr(d, "total_bytes", None))
    except Exception as e:
        print("err", q, str(e)[:100])
lb = api.competition_leaderboard_view(c)
for i, r in enumerate(lb[:12]):
    print(i + 1, r.team_name, r.score)

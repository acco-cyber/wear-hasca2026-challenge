"""Fetch Kaggle competition pages (overview/rules/data) and discussion topics as text.
Read-only; writes text dumps next to this script. Never prints credentials."""
import json, os, sys, inspect
from kaggle.api.kaggle_api_extended import KaggleApi

OUT = os.path.dirname(os.path.abspath(__file__))
api = KaggleApi()
api.authenticate()

comps = sys.argv[1:] or ["3rd-wear-dataset-challenge-hasca-2026", "2nd-wear-dataset-challenge"]

def dump(obj):
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        return [dump(o) for o in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: dump(v) for k, v in vars(obj).items() if not k.startswith("__")}
    return obj

print(inspect.signature(api.competition_list_pages))
print(inspect.signature(api.competition_list_topics))
print(inspect.signature(api.competition_list_topic_messages))
for c in comps:
    try:
        pages = api.competition_list_pages(c)
        d = dump(pages)
        with open(os.path.join(OUT, f"pages_{c}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, default=str, ensure_ascii=False)
        print(c, "pages ok", type(pages), len(pages) if hasattr(pages, "__len__") else "")
    except Exception as e:
        print(c, "pages ERR", repr(e)[:300])
    try:
        topics = api.competition_list_topics(c)
        d = dump(topics)
        with open(os.path.join(OUT, f"topics_{c}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, default=str, ensure_ascii=False)
        print(c, "topics ok", len(topics) if hasattr(topics, "__len__") else "")
    except Exception as e:
        print(c, "topics ERR", repr(e)[:300])

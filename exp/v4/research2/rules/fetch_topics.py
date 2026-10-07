"""Fetch all discussion topic messages for both WEAR competitions (read-only)."""
import json, os
from kaggle.api.kaggle_api_extended import KaggleApi

OUT = os.path.dirname(os.path.abspath(__file__))
api = KaggleApi()
api.authenticate()


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


for comp, tag in [("3rd-wear-dataset-challenge-hasca-2026", "w26"), ("2nd-wear-dataset-challenge", "w25")]:
    topics = json.load(open(os.path.join(OUT, f"topics_{comp}.json"), encoding="utf-8"))["topics"]
    for t in topics:
        tid = t["id"]
        try:
            msgs = dump(api.competition_list_topic_messages(comp, tid, page_size=100))
        except Exception as e:
            print(comp, tid, "ERR", repr(e)[:300])
            continue
        with open(os.path.join(OUT, f"{tag}_topic_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump({"topic": t, "messages": msgs}, f, indent=1, default=str, ensure_ascii=False)
        n = len(msgs.get("messages", msgs)) if isinstance(msgs, dict) else len(msgs)
        print(comp, tid, t["title"][:70], "msgs", n)

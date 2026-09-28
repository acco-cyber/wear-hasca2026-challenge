"""Print the key stdout lines of a Kaggle kernel log (JSON stream format). python log_summary.py <log> [pattern ...]"""
import sys, json
log = json.load(open(sys.argv[1], encoding="utf-8"))
pats = sys.argv[2:] or ["GPU=", "queues", "OOF macroF1", "window-level OOF", "L0 OOF", "window blend + label", "argmax distribution",
                        "ridge r2", "scorer rows", "test linked", "prediction counts", "null share", "total ", "W3 window models |",
                        "[final_s", "[imu_s", "time guard", "error", "Error", "kept"]
for e in log:
    if e.get("stream_name") != "stdout":
        continue
    d = e["data"].rstrip("\n")
    if any(p in d for p in pats):
        print(d[:240])

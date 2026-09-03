#!/bin/bash
# Live progress of whatever runs exist under runs/. Reads history.json per run, so
# it works even while a sweep's own log is still buffered inside a pipe.
#
#   ./watch_ladder.sh          # snapshot
#   watch -n 60 ./watch_ladder.sh
cd "$(dirname "$0")"

if pgrep -f "src.train" >/dev/null 2>&1; then
  echo "STATUS: training active   $(date +%H:%M)"
else
  echo "STATUS: no training process running   $(date +%H:%M)"
fi
echo

.venv/bin/python - <<'PY'
import json, glob, os

def metric_key(row):
    """History rows are keyed by the split that drove selection."""
    for k in ("dev_mAP", "val_mAP"):
        if k in row:
            return k
    return None

rows = []
for h in sorted(glob.glob("runs/*/history.json")):
    run = os.path.basename(os.path.dirname(h))
    try:
        hist = json.load(open(h))
    except (json.JSONDecodeError, OSError):
        continue          # mid-write
    if not hist:
        continue
    key = metric_key(hist[-1])
    if key is None:
        continue
    last = hist[-1]
    best = max(r[key] for r in hist if key in r)
    done = os.path.exists(os.path.join(os.path.dirname(h), "summary.json"))
    rows.append((run, len(hist), last["train_loss"], best, last[key],
                 last.get("lr"), last["epoch_s"], key[:3],
                 "done" if done else "..."))

if not rows:
    print("no runs started yet")
else:
    n_done = sum(1 for r in rows if r[8] == "done")
    print(f"{n_done}/{len(rows)} started runs complete\n")
    print(f"{'run':24s} {'ep':>3} {'loss':>7} {'best':>8} {'last':>7} "
          f"{'lr':>9} {'s/ep':>6} {'sel':>4}  st")
    for r in rows:
        lr = f"{r[5]:9.2e}" if r[5] is not None else "        -"
        print(f"{r[0]:24s} {r[1]:>3} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>7.4f} "
              f"{lr} {r[6]:>6.0f} {r[7]:>4}  {r[8]}")
    print("\nsel = which split the 'best'/'last' columns come from "
          "(dev = honest selection; val = legacy runs)")
PY

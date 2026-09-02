#!/bin/bash
# Live progress of the ablation ladder. Reads history.json per run, so it works
# even while the sweep's own log is still buffered inside a pipe.
#
#   ./watch_ladder.sh          # snapshot
#   watch -n 60 ./watch_ladder.sh
cd "$(dirname "$0")"

if pgrep -f "src.train" >/dev/null 2>&1; then
  echo "STATUS: training active   $(date +%H:%M)"
else
  echo "STATUS: no training process running   $(date +%H:%M)"
fi

TOTAL=12   # 4 arms x 3 seeds
DONE=$(ls runs/*/summary.json 2>/dev/null | grep -c stem2)
echo "complete: ${DONE}/${TOTAL} runs"
echo

.venv/bin/python - <<'PY'
import json, glob, os
rows = []
for h in sorted(glob.glob("runs/*stem2/history.json")):
    run = os.path.basename(os.path.dirname(h))
    hist = json.load(open(h))
    if not hist:
        continue
    last = hist[-1]
    best = max(r["val_mAP"] for r in hist)
    done = os.path.exists(os.path.join(os.path.dirname(h), "summary.json"))
    rows.append((run, len(hist), last["train_loss"], best, last["val_mAP"],
                 last["lr"], last["epoch_s"], "done" if done else "..."))

if not rows:
    print("no runs started yet")
else:
    print(f"{'run':24s} {'ep':>3} {'loss':>7} {'best mAP':>9} {'last':>7} "
          f"{'lr':>9} {'s/ep':>6}  st")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:>3} {r[2]:>7.4f} {r[3]:>9.4f} {r[4]:>7.4f} "
              f"{r[5]:>9.2e} {r[6]:>6.0f}  {r[7]}")
    eta = sum(1 for _ in rows)
    print(f"\n{len(rows)} run(s) started; ~4 min/epoch dual, ~2 min single-stream")
PY

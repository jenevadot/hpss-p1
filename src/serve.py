"""FastAPI serving endpoint.

Design rationale (plan section 6): the inference path is
  raw audio -> STFT -> HPSS (~30 ms CPU) -> mel -> forward
so CPU preprocessing, not the model, dominates. HPSS therefore runs in a process
pool while MPS inference is serialised behind a lock -- there is one GPU and
concurrent submissions contend rather than overlap.

Ray Serve was considered and rejected: splitting CPU-preprocess from GPU-model
deployments is a genuine Serve strength, but running a cluster to serve one model
on one machine is overhead FastAPI avoids. See the plan for the full comparison.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/uvicorn src.serve:app --port 8000
"""
from __future__ import annotations

import asyncio
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config as C
# The SAME feature function used by precompute. Importing it (rather than
# reimplementing) is what prevents train/serve feature skew, which degrades
# predictions silently with no error.
from .features import features_from_path
from .model import HSPPNet

STATE: dict = {}
_MPS_LOCK = asyncio.Lock()


def _load_checkpoint(run_dir: Path):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    arm = ckpt.get("arm", "dual")
    model = HSPPNet(arm=arm)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    thresh_path = run_dir / "thresholds.npy"
    thresholds = (np.load(thresh_path) if thresh_path.exists()
                  else np.full(C.N_CLASSES, 0.5))
    return model, device, arm, thresholds


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_dir = Path(STATE.get("run_dir", C.RUNS_DIR / "dual_seed42"))
    model, device, arm, thresholds = _load_checkpoint(run_dir)
    STATE.update(model=model, device=device, arm=arm, thresholds=thresholds,
                 pool=ProcessPoolExecutor(max_workers=4), run_dir=str(run_dir))
    # Normalisation stats must match training. They are derived from the train
    # split, so they live with the run rather than being recomputed here.
    norm_path = run_dir / "norm.json"
    if norm_path.exists():
        import json
        STATE["norm"] = tuple(json.loads(norm_path.read_text()))
    yield
    STATE["pool"].shutdown()


app = FastAPI(title="AnuraSet HSPP species detector", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in STATE else "loading",
        "arm": STATE.get("arm"),
        "device": str(STATE.get("device")),
        "run_dir": STATE.get("run_dir"),
        "n_classes": C.N_CLASSES,
    }


def _preprocess(path: str):
    x_h, x_p = features_from_path(path)
    return x_h.astype(np.float32), x_p.astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 10):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")

    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop = asyncio.get_running_loop()
        # HPSS is the bottleneck, so it runs off the event loop in a process pool.
        x_h, x_p = await loop.run_in_executor(STATE["pool"], _preprocess, tmp_path)

        norm = STATE.get("norm")
        if norm:
            x_h = (x_h - norm[0]) / norm[1]
            x_p = (x_p - norm[0]) / norm[1]

        device, model, arm = STATE["device"], STATE["model"], STATE["arm"]
        t_h = torch.from_numpy(x_h)[None, None].to(device)
        t_p = torch.from_numpy(x_p)[None, None].to(device)

        # One GPU: serialise forward passes rather than letting them contend.
        async with _MPS_LOCK:
            with torch.no_grad():
                logits = model(t_h, t_p) if arm == "dual" else model(t_h)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    thresholds = STATE["thresholds"]
    detected = [
        {"species": C.SPECIES[i], "probability": round(float(probs[i]), 4),
         "threshold": round(float(thresholds[i]), 3)}
        for i in range(C.N_CLASSES) if probs[i] >= thresholds[i]
    ]
    detected.sort(key=lambda d: -d["probability"])

    order = np.argsort(-probs)[:top_k]
    return {
        "filename": file.filename,
        # Both the binary decisions and the raw scores: thresholds are tuned per
        # class on val, so a caller may reasonably want to re-threshold.
        "detected": detected,
        "top_k": [
            {"species": C.SPECIES[i], "probability": round(float(probs[i]), 4)}
            for i in order
        ],
        "note": ("species with <100 training positives are unreliable: "
                 f"{C.ULTRA_RARE + C.ZERO_POSITIVE}"),
    }

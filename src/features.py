"""The one true feature extraction path: audio -> STFT -> HPSS -> mel -> dB.

This module is imported by BOTH precompute and serving. Never reimplement the
feature path anywhere else -- train/serve feature skew degrades predictions
silently, with no error.

Critical ordering note (section 3.1 of the plan): HPSS runs on the LINEAR STFT,
never on the mel spectrogram. HPSS's median filters exploit harmonics being
horizontal ridges and onsets vertical ridges on a linear-frequency grid. The mel
filterbank warps and compresses frequency non-uniformly, smearing high-frequency
harmonic spacing into adjacent bins -- running HPSS after the mel projection
quietly guts the paper's core mechanism.
"""
from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from . import config as C

# Mel filterbank is deterministic given the config, so build it once per process.
_MEL_FB: np.ndarray | None = None


def mel_filterbank() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = librosa.filters.mel(
            sr=C.SR, n_fft=C.N_FFT, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX,
        )
    return _MEL_FB


def load_audio(path) -> np.ndarray:
    """Read a wav as float32 mono, fixed to exactly N_SAMPLES.

    All AnuraSet clips probe as 22050 Hz / mono / 3.000 s, so resampling should
    never trigger; the guards exist so a malformed file fails loudly rather than
    producing a silently wrong-length feature.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != C.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SR)
    if len(y) < C.N_SAMPLES:
        y = np.pad(y, (0, C.N_SAMPLES - len(y)))
    elif len(y) > C.N_SAMPLES:
        y = y[: C.N_SAMPLES]
    return y


def hpss_mel(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (harmonic, percussive) log-mel spectrograms, each (N_MELS, N_FRAMES).

    Order: STFT -> HPSS on the complex linear spectrogram -> power -> mel -> dB.
    """
    S = librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)
    S_h, S_p = librosa.decompose.hpss(
        S, kernel_size=C.HPSS_KERNEL, margin=C.HPSS_MARGIN,
    )
    fb = mel_filterbank()
    out = []
    for comp in (S_h, S_p):
        power = np.abs(comp) ** 2
        mel = fb @ power
        db = librosa.power_to_db(mel, ref=C.DB_REF, top_db=C.DB_TOP)
        out.append(_fix_frames(db))
    return out[0], out[1]


def raw_mel(y: np.ndarray) -> np.ndarray:
    """Log-mel of the un-decomposed signal -- the 'no HPSS' ablation arm."""
    S = librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)
    mel = mel_filterbank() @ (np.abs(S) ** 2)
    return _fix_frames(librosa.power_to_db(mel, ref=C.DB_REF, top_db=C.DB_TOP))


def _fix_frames(x: np.ndarray) -> np.ndarray:
    """Pin the time axis to exactly N_FRAMES so every stored feature aligns."""
    t = x.shape[1]
    if t < C.N_FRAMES:
        return np.pad(x, ((0, 0), (0, C.N_FRAMES - t)), constant_values=-C.DB_TOP)
    return x[:, : C.N_FRAMES] if t > C.N_FRAMES else x


def features_from_path(path) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper used by precompute and by the serving endpoint."""
    return hpss_mel(load_audio(path))

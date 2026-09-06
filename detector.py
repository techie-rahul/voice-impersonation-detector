#!/usr/bin/env python3
"""
detector.py -- voice-cloning / spoof detection using a pretrained AASIST model.

Thin wrapper around the vendored clovaai/aasist network (``ml/aasist``) and its
committed pretrained weights (``ml/aasist/models/weights/AASIST.pth``), trained on
ASVspoof 2019 LA.

The model consumes ~4 s of raw 16 kHz waveform and returns two logits
``(spoof, bonafide)``. Following the AASIST convention (see ``ml/aasist/main.py``),
the countermeasure score is ``logits[:, 1]`` -- higher means "more likely genuine".

Usage
-----
    python detector.py path/to/audio.flac [more_audio.wav ...]
    python detector.py --self-test                 # known bonafide + spoof sample
    python detector.py --device cpu audio.flac     # force a device
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
AASIST_DIR = REPO_ROOT / "ml" / "aasist"
AASIST_CONFIG = AASIST_DIR / "config" / "AASIST.conf"
AASIST_WEIGHTS = AASIST_DIR / "models" / "weights" / "AASIST.pth"

# In-repo symlink -> ASVspoof2019 root. Note the doubled "LA/LA" from extraction.
DATASET_LA = REPO_ROOT / "data" / "asvspoof2019" / "LA" / "LA"
TRAIN_FLAC = DATASET_LA / "ASVspoof2019_LA_train" / "flac"
TRAIN_PROTOCOL = (
    DATASET_LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
)

SAMPLE_RATE = 16_000
NUM_SAMPLES = 64_600  # ~4.04 s -- AASIST's fixed input length

# Make ``from models.AASIST import Model`` resolve regardless of the caller's cwd.
if str(AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_DIR))


def select_device(requested: str = "auto") -> torch.device:
    """Resolve a torch device, preferring CUDA, then Apple MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad(x: np.ndarray, max_len: int = NUM_SAMPLES) -> np.ndarray:
    """Deterministically crop/tile a 1-D signal to ``max_len`` samples.

    Mirrors ``ml/aasist/data_utils.py:pad`` (front crop, tile-repeat for short
    clips) so inference matches how AASIST was evaluated.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def load_waveform(path: Path | str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float waveform at ``target_sr``."""
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:  # stereo/multi-channel -> mono
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float64)
    if sr != target_sr:
        import librosa  # imported lazily; only needed for non-16 kHz input

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


class AASISTDetector:
    """Pretrained AASIST spoof/bonafide classifier."""

    LABELS = {0: "spoof", 1: "bonafide"}

    def __init__(
        self,
        weights: Path | str = AASIST_WEIGHTS,
        config: Path | str = AASIST_CONFIG,
        device: str = "auto",
    ) -> None:
        from models.AASIST import Model  # vendored: ml/aasist/models/AASIST.py

        with open(config) as fh:
            self.config = json.load(fh)

        self.device = select_device(device)
        self.model = Model(self.config["model_config"]).to(self.device)
        state_dict = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def score(self, audio_path: Path | str) -> dict:
        """Classify one audio file.

        Returns a dict with the predicted label, the raw AASIST countermeasure
        score (``logits[1]``; higher = more genuine) and softmax probabilities.
        """
        wav = pad(load_waveform(audio_path))
        x = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

        _, logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(logits, dim=1).item())

        return {
            "file": str(audio_path),
            "prediction": self.LABELS[pred],
            "cm_score": float(logits[0, 1].item()),
            "p_bonafide": float(probs[1].item()),
            "p_spoof": float(probs[0].item()),
        }

    def score_many(self, audio_paths) -> list[dict]:
        return [self.score(p) for p in audio_paths]


# --- Module-level convenience API -------------------------------------------------

_DEFAULT_DETECTOR: "AASISTDetector | None" = None


def detect_synthetic(audio_path: Path | str) -> dict:
    """Classify one audio file as synthetic/spoof vs. genuine/bonafide.

    Lightweight wrapper around :class:`AASISTDetector`. The first call builds a
    single shared detector (AASIST weight loading is expensive); every later call
    reuses that same instance. For an explicit device or batch scoring, use
    :class:`AASISTDetector` directly.

    Returns the same dict as :meth:`AASISTDetector.score`::

        {
            "file":       str,
            "prediction": "spoof" | "bonafide",   # argmax of the two logits
            "cm_score":   float,   # bonafide-class logit, unbounded; higher = more genuine
            "p_bonafide": float,   # softmax probability in [0, 1]
            "p_spoof":    float,   # softmax probability in [0, 1]
        }
    """
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = AASISTDetector()
    return _DEFAULT_DETECTOR.score(audio_path)


def _format_row(res: dict) -> str:
    return (
        f"  {res['prediction']:>8}  "
        f"cm_score={res['cm_score']:+8.4f}  "
        f"P(bonafide)={res['p_bonafide']:.4f}  "
        f"P(spoof)={res['p_spoof']:.4f}  "
        f"{Path(res['file']).name}"
    )


def _lookup_sample(prediction_wanted: str) -> Path:
    """Return the first train-set clip with the requested label from the protocol."""
    if not TRAIN_PROTOCOL.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {TRAIN_PROTOCOL}\n"
            "The in-repo dataset symlink (data/asvspoof2019) may be missing."
        )
    with open(TRAIN_PROTOCOL) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 5 and parts[4] == prediction_wanted:
                return TRAIN_FLAC / f"{parts[1]}.flac"
    raise LookupError(f"No '{prediction_wanted}' entry found in {TRAIN_PROTOCOL}")


def run_self_test(device: str) -> int:
    """Score one known-bonafide and one known-spoof training clip."""
    cases = [("bonafide", _lookup_sample("bonafide")), ("spoof", _lookup_sample("spoof"))]

    print(f"Loading AASIST (weights: {AASIST_WEIGHTS.relative_to(REPO_ROOT)}) ...")
    detector = AASISTDetector(device=device)
    print(f"Device: {detector.device}\n")

    ok = True
    for expected, path in cases:
        res = detector.score(path)
        verdict = "PASS" if res["prediction"] == expected else "FAIL"
        ok &= res["prediction"] == expected
        print(f"[{verdict}] expected {expected}")
        print(_format_row(res))
        print()

    print("Self-test:", "all samples classified correctly." if ok else "MISMATCH -- see above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="*", help="audio file(s) to classify")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--self-test", action="store_true", help="score a known bonafide + spoof sample")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(args.device)

    if not args.audio:
        parser.error("provide at least one audio file, or use --self-test")

    detector = AASISTDetector(device=args.device)
    print(f"Device: {detector.device}\n")
    for path in args.audio:
        print(_format_row(detector.score(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

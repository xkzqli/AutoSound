from pathlib import Path
from collections import Counter
import hashlib
import json
import math

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Engine Issue", "Non engine issue", "Normal engine"]
SR = 16000
EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
VERSION = "spectral-v1"


def extract_features(path):
    # Limit processing to the first 30 seconds.
    with sf.SoundFile(str(path)) as file:
        original_sr = file.samplerate
        audio = file.read(
            frames=int(original_sr * 30),
            dtype="float32", always_2d=True
        )

    audio = audio.mean(axis=1)
    if len(audio) < original_sr * 0.25:
        raise ValueError("Recording shorter than 0.25 seconds")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-7:
        raise ValueError("Silent or invalid audio")

    divisor = math.gcd(original_sr, SR)
    audio = resample_poly(
        audio, SR // divisor, original_sr // divisor
    )
    audio = audio - audio.mean()
    peak = np.max(np.abs(audio))
    if peak < 1e-7:
        raise ValueError("No usable varying signal")
    audio = audio / peak

    # Analyze non-overlapping two-second windows.
    edges = np.geomspace(40, SR / 2, 33)
    windows = []
    for start in range(0, len(audio), SR * 2):
        part = audio[start:start + SR * 2]
        if len(part) < SR // 4:
            continue
        frequencies, power = welch(part, fs=SR, nperseg=1024)
        bands = np.array([
            power[(frequencies >= lo) & (frequencies < hi)].sum()
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        bands = np.log10(bands / (bands.sum() + 1e-12) + 1e-10)
        zcr = np.mean(np.diff(np.signbit(part)) != 0)
        rms = np.sqrt(np.mean(part ** 2))
        windows.append(np.r_[bands, zcr, rms])

    values = np.asarray(windows)
    return np.r_[values.mean(axis=0), values.std(axis=0)]


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    features, labels, paths = [], [], []
    seen, skipped = {}, []

    for label in CLASSES:
        for path in sorted((ROOT / "data/Audio" / label).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(
                        f"Identical audio file has conflicting labels: {relative}"
                    )
                skipped.append({"path": relative, "reason": "Exact duplicate"})
                continue

            try:
                vector = extract_features(path)
            except Exception as error:
                skipped.append({"path": relative, "reason": str(error)})
                print("SKIP:", relative, error, flush=True)
                continue

            seen[digest] = label
            features.append(vector)
            labels.append(label)
            paths.append(relative)
            print(f"Processed {len(features)}: {relative}", flush=True)

    X = np.asarray(features)
    y = np.asarray(labels)
    print("\nUsable recordings:", dict(Counter(labels)), flush=True)

    # Fixed split: do not change the seed to chase a better score.
    train_ids, test_ids = train_test_split(
        np.arange(len(y)), test_size=0.25,
        stratify=y, random_state=42
    )

    model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X[train_ids], y[train_ids])
    predictions = model.predict(X[test_ids])

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X[train_ids], y[train_ids])
    baseline_predictions = baseline.predict(X[test_ids])

    metrics = {
        "feature_version": VERSION,
        "usable_recordings": len(y),
        "train_recordings": len(train_ids),
        "test_recordings": len(test_ids),
        "accuracy": float(accuracy_score(y[test_ids], predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[test_ids], predictions)
        ),
        "macro_f1": float(
            f1_score(y[test_ids], predictions, average="macro")
        ),
        "majority_baseline_accuracy": float(
            accuracy_score(y[test_ids], baseline_predictions)
        ),
        "class_order": CLASSES,
        "confusion_matrix": confusion_matrix(
            y[test_ids], predictions, labels=CLASSES
        ).tolist(),
        "classification_report": classification_report(
            y[test_ids], predictions, labels=CLASSES,
            output_dict=True, zero_division=0
        ),
        "limitation": (
            "Exploratory recording-level split; vehicle/source identity "
            "is unknown. Exact file deduplication does not remove "
            "re-encoded copies or clips from the same original recording."
        )
    }

    joblib.dump({
        "model": model,
        "feature_version": VERSION,
        "sample_rate": SR,
        "reference_features": X[train_ids],
        "reference_labels": y[train_ids],
        "reference_paths": [paths[i] for i in train_ids]
    }, ROOT / "models/autosound_v1.joblib")

    outputs = {
        "metrics.json": metrics,
        "skipped_files.json": skipped,
        "split.json": {
            "train": [paths[i] for i in train_ids],
            "test": [paths[i] for i in test_ids]
        },
        "test_predictions.json": [
            {"path": paths[i], "actual": str(y[i]), "predicted": str(p)}
            for i, p in zip(test_ids, predictions)
        ]
    }
    for filename, content in outputs.items():
        (ROOT / "results" / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print("\nTRAINING COMPLETE")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Majority baseline: {metrics['majority_baseline_accuracy']:.1%}")
    print(classification_report(
        y[test_ids], predictions, labels=CLASSES, zero_division=0
    ))
    print("Saved model: models/autosound_v1.joblib")
    print("Saved evaluation: results/metrics.json")


if __name__ == "__main__":
    main()

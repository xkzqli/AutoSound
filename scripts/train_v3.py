from pathlib import Path
import json
import sys

import joblib
import librosa
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import extract_features as old_features

VERSION = "mfcc-v3"

def summarize(matrix):
    return np.concatenate([
        np.mean(matrix, axis=1),
        np.std(matrix, axis=1),
        np.percentile(matrix, 10, axis=1),
        np.percentile(matrix, 90, axis=1)
    ])

def extract_features(path):
    # Same 30-second limit as V1.
    y, sr = librosa.load(str(path), sr=16000, mono=True, duration=30)
    if len(y) < 4000 or not np.isfinite(y).all():
        raise ValueError("Invalid or too-short audio")
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak < 1e-7:
        raise ValueError("Silent audio")
    y = y / peak

    power = np.abs(librosa.stft(
        y, n_fft=1024, hop_length=256
    )) ** 2
    mel = librosa.feature.melspectrogram(
        S=power, sr=sr, n_mels=64, fmin=40, fmax=8000
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    spectral = np.vstack([
        librosa.feature.spectral_centroid(S=np.sqrt(power), sr=sr),
        librosa.feature.spectral_bandwidth(S=np.sqrt(power), sr=sr),
        librosa.feature.spectral_rolloff(S=np.sqrt(power), sr=sr),
        librosa.feature.spectral_flatness(S=np.sqrt(power)),
        librosa.feature.zero_crossing_rate(
            y, frame_length=1024, hop_length=256
        ),
        librosa.feature.rms(S=np.sqrt(power), frame_length=1024)
    ])

    return np.nan_to_num(np.concatenate([
        summarize(mfcc), summarize(delta), summarize(delta2),
        summarize(logmel), summarize(spectral)
    ]))

def main():
    split = json.loads(
        (ROOT / "results/split.json").read_text(encoding="utf-8")
    )
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Train/test overlap")

    train_paths = split["train"]
    labels = np.array([Path(p).parent.name for p in train_paths])
    old = joblib.load(ROOT / "models/autosound_v1.joblib")
    lookup = dict(zip(old["reference_paths"], old["reference_features"]))
    basic = np.asarray([lookup[p] for p in train_paths])

    cache_path = ROOT / "models/mfcc_v3_cache.joblib"
    cache = joblib.load(cache_path) if cache_path.exists() else {}

    def get_features(relative):
        path = ROOT / relative
        stat = path.stat()
        key = (relative, stat.st_size, stat.st_mtime_ns, VERSION)
        if key not in cache:
            cache[key] = extract_features(path)
        return cache[key]

    advanced = []
    for i, relative in enumerate(train_paths, 1):
        advanced.append(get_features(relative))
        if i % 10 == 0 or i == len(train_paths):
            print(f"Training features: {i}/{len(train_paths)}", flush=True)
            joblib.dump(cache, cache_path)

    advanced = np.asarray(advanced)
    feature_sets = {
        "MFCC": advanced,
        "Combined": np.hstack([basic, advanced])
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    best_score, best_model, best_set = -1, None, None

    for feature_name, X in feature_sets.items():
        models = {
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "SVM_C1": make_pipeline(
                StandardScaler(),
                SVC(C=1, class_weight="balanced")
            ),
            "SVM_C10": make_pipeline(
                StandardScaler(),
                SVC(C=10, class_weight="balanced")
            )
        }
        for name, model in models.items():
            values = cross_val_score(
                model, X, labels, cv=cv, scoring="f1_macro",
                n_jobs=1, error_score="raise"
            )
            mean = float(values.mean())
            scores.append({
                "features": feature_name, "model": name,
                "cv_macro_f1": mean,
                "fold_scores": values.tolist()
            })
            print(
                f"{feature_name} + {name}: CV Macro-F1 {mean:.3f}",
                flush=True
            )
            if mean > best_score:
                best_score = mean
                best_model = clone(model)
                best_set = feature_name

    X = feature_sets[best_set]
    best_model.fit(X, labels)
    print(f"\nSelected features: {best_set}", flush=True)

    test = []
    actual = []
    for i, relative in enumerate(split["test"], 1):
        vector = get_features(relative)
        if best_set == "Combined":
            vector = np.r_[old_features(ROOT / relative), vector]
        test.append(vector)
        actual.append(Path(relative).parent.name)
        if i % 10 == 0:
            print(f"Test features: {i}/{len(split['test'])}", flush=True)

    prediction = best_model.predict(np.asarray(test))
    metrics = {
        "feature_version": VERSION,
        "feature_set": best_set,
        "cross_validation": scores,
        "train_recordings": len(labels),
        "test_recordings": len(actual),
        "accuracy": float(accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, average="macro")),
        "classification_report": classification_report(
            actual, prediction, output_dict=True, zero_division=0
        ),
        "limitation": (
            "Exploratory results on a previously inspected test set; "
            "source independence and dataset labels remain unverified."
        )
    }

    joblib.dump({
        "model": best_model,
        "feature_version": VERSION,
        "feature_set": best_set,
        "reference_features": X,
        "reference_labels": labels,
        "reference_paths": train_paths
    }, ROOT / "models/autosound_v3.joblib")
    joblib.dump(cache, cache_path)
    (ROOT / "results/metrics_v3.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nV3 COMPLETE")
    print(f"Best CV Macro-F1: {best_score:.3f}")
    print(f"Test accuracy: {metrics['accuracy']:.1%}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.3f}")
    print("V1: accuracy 68.1%, Macro-F1 0.673")
    print(classification_report(actual, prediction, zero_division=0))
    print("Old model and web app are unchanged.")

if __name__ == "__main__":
    main()

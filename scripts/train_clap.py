from pathlib import Path
import hashlib
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import joblib
import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "laion/clap-htsat-unfused"
VERSION = "clap-10sec-mean-v1"
SR = 48000

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    print("Loading cached CLAP model...", flush=True)
    processor = ClapProcessor.from_pretrained(
        MODEL_ID, local_files_only=True
    )
    encoder = ClapModel.from_pretrained(
        MODEL_ID, local_files_only=True
    ).eval()
    revision = getattr(encoder.config, "_commit_hash", None)

    cache_path = ROOT / "models/clap_features_cache.joblib"
    cache = joblib.load(cache_path) if cache_path.exists() else {}

    def features(relative):
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (VERSION, MODEL_ID, revision, digest)
        if key in cache:
            return cache[key]

        y, _ = librosa.load(
            str(path), sr=SR, mono=True, duration=30
        )
        if len(y) < SR // 4 or not np.isfinite(y).all():
            raise ValueError(f"Invalid audio: {relative}")
        if np.max(np.abs(y)) < 1e-7:
            raise ValueError(f"Silent audio: {relative}")

        # Separate windows stay together within their original recording.
        vectors = []
        for start in range(0, len(y), SR * 10):
            chunk = y[start:start + SR * 10]
            if len(chunk) < SR // 4:
                continue
            inputs = processor(
                audios=chunk,
                sampling_rate=SR,
                return_tensors="pt"
            )
            with torch.inference_mode():
                embedding = encoder.get_audio_features(**inputs)
            vector = embedding.cpu().numpy().reshape(-1)
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            vectors.append(vector)

        result = np.mean(vectors, axis=0)
        result = result / max(np.linalg.norm(result), 1e-12)
        cache[key] = result
        joblib.dump(cache, cache_path)
        return result

    split = json.loads(
        (ROOT / "results/split.json").read_text(encoding="utf-8")
    )
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Train/test overlap")

    train_paths = split["train"]
    X = []
    for i, relative in enumerate(train_paths, 1):
        X.append(features(relative))
        print(
            f"CLAP training features: {i}/{len(train_paths)}",
            flush=True
        )
    X = np.asarray(X)
    y = np.array([Path(p).parent.name for p in train_paths])

    candidates = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1, max_iter=3000,
                class_weight="balanced", random_state=42
            )
        ),
        "SVM_C1": make_pipeline(
            StandardScaler(),
            SVC(C=1, class_weight="balanced")
        ),
        "SVM_C10": make_pipeline(
            StandardScaler(),
            SVC(C=10, class_weight="balanced")
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    scores = {}
    for name, candidate in candidates.items():
        values = cross_val_score(
            candidate, X, y, cv=cv, scoring="f1_macro",
            n_jobs=1, error_score="raise"
        )
        scores[name] = {
            "mean_macro_f1": float(values.mean()),
            "fold_scores": values.tolist()
        }
        print(
            f"{name}: CV Macro-F1 {values.mean():.3f}",
            flush=True
        )

    winner = max(
        scores, key=lambda name: scores[name]["mean_macro_f1"]
    )
    classifier = clone(candidates[winner])
    classifier.fit(X, y)
    print(f"\nSelected by training CV: {winner}", flush=True)

    test_paths = split["test"]
    test_X = []
    for i, relative in enumerate(test_paths, 1):
        test_X.append(features(relative))
        print(
            f"CLAP test features: {i}/{len(test_paths)}",
            flush=True
        )

    actual = np.array([Path(p).parent.name for p in test_paths])
    prediction = classifier.predict(np.asarray(test_X))
    classes = list(classifier.classes_)

    metrics = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "feature_version": VERSION,
        "selected_classifier": winner,
        "cross_validation": scores,
        "train_recordings": len(train_paths),
        "test_recordings": len(test_paths),
        "accuracy": float(accuracy_score(actual, prediction)),
        "macro_f1": float(
            f1_score(actual, prediction, average="macro")
        ),
        "class_order": classes,
        "confusion_matrix": confusion_matrix(
            actual, prediction, labels=classes
        ).tolist(),
        "classification_report": classification_report(
            actual, prediction, output_dict=True, zero_division=0
        ),
        "limitation": (
            "Exploratory comparison on a repeatedly inspected test set. "
            "Dataset labels, source independence, and overlap with "
            "CLAP pretraining data have not been verified."
        )
    }

    joblib.dump({
        "model": classifier,
        "encoder_id": MODEL_ID,
        "encoder_revision": revision,
        "feature_version": VERSION,
        "sample_rate": SR,
        "reference_features": X,
        "reference_labels": y,
        "reference_paths": train_paths
    }, ROOT / "models/autosound_clap.joblib")

    (ROOT / "results/metrics_clap.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ROOT / "results/test_predictions_clap.json").write_text(
        json.dumps([
            {
                "path": path,
                "actual": str(label),
                "predicted": str(pred)
            }
            for path, label, pred in zip(
                test_paths, actual, prediction
            )
        ], indent=2),
        encoding="utf-8"
    )

    print("\nCLAP COMPLETE")
    print(f"Classifier: {winner}")
    print(
        f"Best CV Macro-F1: "
        f"{scores[winner]['mean_macro_f1']:.3f}"
    )
    print(f"Test accuracy: {metrics['accuracy']:.1%}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.3f}")
    print("V3 reference: accuracy 73.6%, Macro-F1 0.729")
    print(classification_report(actual, prediction, zero_division=0))
    print("V3 and the web app are unchanged.")

if __name__ == "__main__":
    main()

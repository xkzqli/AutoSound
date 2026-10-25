from pathlib import Path
import json
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import extract_features, CLASSES, VERSION

def main():
    old = joblib.load(ROOT / "models/autosound_v1.joblib")
    if old["feature_version"] != VERSION:
        raise ValueError("Feature version mismatch")

    X = np.asarray(old["reference_features"])
    y = np.asarray(old["reference_labels"])

    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced",
            random_state=42, n_jobs=1
        ),
        "ScaledSVM": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", class_weight="balanced")
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    print("Comparing models on TRAINING recordings only...", flush=True)

    for name, model in candidates.items():
        values = cross_val_score(
            model, X, y, cv=cv,
            scoring="f1_macro", n_jobs=1, error_score="raise"
        )
        scores[name] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std()),
            "fold_scores": values.tolist()
        }
        print(
            f"{name}: Macro-F1 = {values.mean():.3f} "
            f"(fold SD {values.std():.3f})",
            flush=True
        )

    best_name = max(scores, key=lambda name: scores[name]["mean_macro_f1"])
    best = clone(candidates[best_name])
    best.fit(X, y)
    print(f"\nSelected by cross-validation: {best_name}", flush=True)

    split = json.loads(
        (ROOT / "results/split.json").read_text(encoding="utf-8")
    )
    test_paths = split["test"]
    if set(test_paths) & set(old["reference_paths"]):
        raise ValueError("Train/test paths overlap")

    test_features = []
    test_labels = []
    for index, relative in enumerate(test_paths, 1):
        path = ROOT / relative
        test_features.append(extract_features(path))
        test_labels.append(path.parent.name)
        print(f"Reading test audio {index}/{len(test_paths)}", flush=True)

    prediction = best.predict(np.asarray(test_features))
    metrics = {
        "selected_model": best_name,
        "feature_version": VERSION,
        "cross_validation": scores,
        "train_recordings": len(y),
        "test_recordings": len(test_labels),
        "accuracy": float(accuracy_score(test_labels, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, prediction)
        ),
        "macro_f1": float(
            f1_score(test_labels, prediction, average="macro")
        ),
        "class_order": CLASSES,
        "confusion_matrix": confusion_matrix(
            test_labels, prediction, labels=CLASSES
        ).tolist(),
        "limitation": (
            "Exploratory comparison on a previously inspected test set. "
            "Vehicle/source identity and label provenance remain unverified. "
            "Exact deduplication does not exclude near-duplicate recordings."
        )
    }

    updated = dict(old)
    updated["model"] = best
    updated["selected_model"] = best_name
    joblib.dump(updated, ROOT / "models/autosound_v2.joblib")
    (ROOT / "results/metrics_v2.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ROOT / "results/test_predictions_v2.json").write_text(
        json.dumps([
            {"path": path, "actual": actual, "predicted": str(pred)}
            for path, actual, pred in zip(
                test_paths, test_labels, prediction
            )
        ], indent=2),
        encoding="utf-8"
    )

    print("\nCOMPARISON COMPLETE")
    print(f"Selected model: {best_name}")
    print(f"V2 accuracy: {metrics['accuracy']:.1%}")
    print(f"V2 balanced accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"V2 Macro-F1: {metrics['macro_f1']:.3f}")
    print("V1 reference: accuracy 68.1%, Macro-F1 0.673")
    print(classification_report(
        test_labels, prediction, labels=CLASSES, zero_division=0
    ))
    print("V1 and the web app have NOT been changed.")

if __name__ == "__main__":
    main()

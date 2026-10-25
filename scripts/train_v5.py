from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import extract_features as basic_features
from train_v3 import VERSION as MFCC_VERSION

def main():
    v3 = joblib.load(ROOT / "models/autosound_v3.joblib")
    clap = joblib.load(ROOT / "models/autosound_clap.joblib")
    split = json.loads(
        (ROOT / "results/split.json").read_text(encoding="utf-8")
    )

    paths = split["train"]
    if set(paths) & set(split["test"]):
        raise ValueError("Train/test overlap")

    v3_lookup = dict(zip(
        v3["reference_paths"], v3["reference_features"]
    ))
    clap_lookup = dict(zip(
        clap["reference_paths"], clap["reference_features"]
    ))

    labels = np.array([Path(p).parent.name for p in paths])
    for bundle in [v3, clap]:
        label_lookup = dict(zip(
            bundle["reference_paths"], bundle["reference_labels"]
        ))
        if any(label_lookup.get(p) != label for p, label in zip(paths, labels)):
            raise ValueError("Training paths or labels do not match")

    A = np.asarray([v3_lookup[p] for p in paths])
    B = np.asarray([clap_lookup[p] for p in paths])
    fused = np.hstack([A, B])

    feature_sets = {"V3": A, "CLAP": B, "Fusion": fused}

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    # Select each model's own feature block.
    a_columns = list(range(A.shape[1]))
    b_columns = list(range(A.shape[1], fused.shape[1]))

    v3_branch = make_pipeline(
        ColumnTransformer(
            [("v3_features", "passthrough", a_columns)],
            remainder="drop"
        ),
        clone(v3["model"])
    )
    clap_branch = make_pipeline(
        ColumnTransformer(
            [("clap_features", "passthrough", b_columns)],
            remainder="drop"
        ),
        CalibratedClassifierCV(
            estimator=clone(clap["model"]),
            method="sigmoid",
            cv=3
        )
    )

    candidates = {
        "Previous_Fusion": (
            "Fusion",
            ExtraTreesClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    }

    # Probability calibration is fitted inside each training fold.
    for weights in [(1, 1), (1, 2), (2, 1)]:
        name = f"Voting_V3_{weights[0]}_CLAP_{weights[1]}"
        candidates[name] = (
            "Fusion",
            VotingClassifier(
                estimators=[
                    ("v3", clone(v3_branch)),
                    ("clap", clone(clap_branch))
                ],
                voting="soft",
                weights=list(weights),
                n_jobs=1
            )
        )

    for fraction in [0.3, 0.7]:
        for leaf in [1, 2]:
            name = f"Fusion_Trees_features{fraction}_leaf{leaf}"
            candidates[name] = (
                "Fusion",
                ExtraTreesClassifier(
                    n_estimators=700,
                    max_features=fraction,
                    min_samples_leaf=leaf,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    results = {}

    print("Comparing final candidates using training data only...", flush=True)
    for name, (feature_set, model) in candidates.items():
        values = cross_val_score(
            model, feature_sets[feature_set], labels,
            cv=cv, scoring="f1_macro",
            n_jobs=1, error_score="raise"
        )
        results[name] = {
            "feature_set": feature_set,
            "cv_macro_f1": float(values.mean()),
            "fold_scores": values.tolist()
        }
        print(
            f"{name}: CV Macro-F1 {values.mean():.3f}",
            flush=True
        )

    winner = max(
        results, key=lambda name: results[name]["cv_macro_f1"]
    )
    feature_set, candidate = candidates[winner]
    model = clone(candidate)
    model.fit(feature_sets[feature_set], labels)
    print(f"\nSelected by training CV: {winner}", flush=True)

    # Read previously computed test features from caches.
    mfcc_cache = None
    clap_cache = None
    if feature_set in {"V3", "Fusion"}:
        mfcc_cache = joblib.load(ROOT / "models/mfcc_v3_cache.joblib")
    if feature_set in {"CLAP", "Fusion"}:
        clap_cache = joblib.load(ROOT / "models/clap_features_cache.joblib")

    def test_vector(relative):
        path = ROOT / relative
        a, b = None, None

        if feature_set in {"V3", "Fusion"}:
            stat = path.stat()
            key = (
                relative, stat.st_size, stat.st_mtime_ns,
                MFCC_VERSION
            )
            if key not in mfcc_cache:
                raise ValueError(
                    f"MFCC cache missing or file changed: {relative}"
                )
            advanced = mfcc_cache[key]
            a = (
                np.r_[basic_features(path), advanced]
                if v3["feature_set"] == "Combined" else advanced
            )

        if feature_set in {"CLAP", "Fusion"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            key = (
                clap["feature_version"], clap["encoder_id"],
                clap.get("encoder_revision"), digest
            )
            if key not in clap_cache:
                raise ValueError(
                    f"CLAP cache missing or file changed: {relative}"
                )
            b = clap_cache[key]

        if feature_set == "V3":
            return a
        if feature_set == "CLAP":
            return b
        return np.r_[a, b]

    test_paths = split["test"]
    test_X = np.asarray([test_vector(p) for p in test_paths])
    actual = np.array([Path(p).parent.name for p in test_paths])
    prediction = model.predict(test_X)

    accuracy = float(accuracy_score(actual, prediction))
    macro_f1 = float(f1_score(actual, prediction, average="macro"))

    bundle = {
        "model": model,
        "feature_set": feature_set,
        "selected_candidate": winner,
        "feature_version": "ensemble-v5",
        "v3_feature_set": v3["feature_set"],
        "clap_encoder_id": clap["encoder_id"],
        "clap_encoder_revision": clap.get("encoder_revision"),
        "clap_feature_version": clap["feature_version"],
        "feature_dimensions": {"v3": A.shape[1], "clap": B.shape[1]},
        "reference_features": feature_sets[feature_set],
        "reference_labels": labels,
        "reference_paths": paths
    }
    joblib.dump(bundle, ROOT / "models/autosound_v5.joblib")

    report = {
        "selected_candidate": winner,
        "feature_set": feature_set,
        "cross_validation": results,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_recordings": len(paths),
        "test_recordings": len(test_paths),
        "classification_report": classification_report(
            actual, prediction, output_dict=True, zero_division=0
        ),
        "limitation": (
            "Exploratory evaluation after repeated development on the same "
            "test set. Source independence, label quality, and pretraining "
            "overlap have not been established."
        )
    }
    (ROOT / "results/metrics_v5.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ROOT / "results/test_predictions_v5.json").write_text(
        json.dumps([
            {"path": p, "actual": str(a), "predicted": str(b)}
            for p, a, b in zip(test_paths, actual, prediction)
        ], indent=2),
        encoding="utf-8"
    )

    print("\nV5 COMPARISON COMPLETE")
    print(f"Selected: {winner}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Correct: {int(np.sum(actual == prediction))}/{len(actual)}")
    print(f"Macro-F1: {macro_f1:.3f}")
    print(f"Difference from V3: {(accuracy - 53/72)*100:+.2f} percentage points")
    print(classification_report(actual, prediction, zero_division=0))
    print("Saved: models/autosound_v5.joblib")
    print("Existing web app and older models were not changed.")

if __name__ == "__main__":
    main()

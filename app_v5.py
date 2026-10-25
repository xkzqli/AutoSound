from pathlib import Path
import io
import json
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from train import extract_features as basic_features
from train_v3 import extract_features as advanced_features

VERSION = "ensemble-v5"

@st.cache_resource
def load_clap_encoder(model_id, revision):
    import torch
    from transformers import ClapModel, ClapProcessor

    torch.set_num_threads(4)
    options = {"local_files_only": True}
    if revision:
        options["revision"] = revision

    processor = ClapProcessor.from_pretrained(model_id, **options)
    encoder = ClapModel.from_pretrained(model_id, **options).eval()
    return processor, encoder

def extract_features(path):
    import librosa
    import torch

    mode = bundle["feature_set"]
    parts = []

    if mode in {"V3", "Fusion"}:
        advanced = advanced_features(path)
        if bundle["v3_feature_set"] == "Combined":
            advanced = np.r_[basic_features(path), advanced]
        parts.append(advanced)

    if mode in {"CLAP", "Fusion"}:
        processor, encoder = load_clap_encoder(
            bundle["clap_encoder_id"],
            bundle.get("clap_encoder_revision")
        )
        audio, _ = librosa.load(
            str(path), sr=48000, mono=True, duration=30
        )
        if len(audio) < 12000 or not np.isfinite(audio).all():
            raise ValueError("Invalid or too-short recording")
        if np.max(np.abs(audio)) < 1e-7:
            raise ValueError("Silent recording")

        vectors = []
        for start in range(0, len(audio), 480000):
            chunk = audio[start:start + 480000]
            if len(chunk) < 12000:
                continue
            inputs = processor(
                audios=chunk, sampling_rate=48000,
                return_tensors="pt"
            )
            with torch.inference_mode():
                embedding = encoder.get_audio_features(**inputs)
            vector = embedding.cpu().numpy().reshape(-1)
            vectors.append(
                vector / max(np.linalg.norm(vector), 1e-12)
            )

        vector = np.mean(vectors, axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        parts.append(vector)

    if not parts:
        raise ValueError("Unsupported feature set")
    return np.concatenate(parts)

st.set_page_config(page_title="AutoSound", page_icon="🚗", layout="wide")
st.title("AutoSound")
st.write("Vehicle audio classification and reference matching")
st.warning(
    "Experimental prototype. Predictions use dataset labels and do not "
    "establish whether a vehicle is safe or identify a specific failed part."
)

@st.cache_resource
def load_bundle():
    bundle = joblib.load(ROOT / "models/autosound_v5.joblib")
    if bundle["feature_version"] != VERSION:
        raise ValueError("Feature version mismatch. Model and code must match.")
    scaler = StandardScaler().fit(bundle["reference_features"])
    references = scaler.transform(bundle["reference_features"])
    return bundle, scaler, references

try:
    bundle, scaler, references = load_bundle()
except Exception as error:
    st.error(f"Could not load the local model: {error}")
    st.stop()

display_names = {
    "Engine Issue": "Engine issue",
    "Non engine issue": "Non-engine issue",
    "Normal engine": "Normal engine"
}

st.caption(
    "Analyzes the first 30 seconds. Reference matches use training recordings "
    "only. Testing a training recording is not an independent evaluation."
)

uploaded = st.file_uploader(
    "Upload a recording", type=["mp3", "wav", "flac", "ogg"]
)

if uploaded is not None:
    raw = uploaded.getvalue()
    if len(raw) > 20 * 1024 * 1024:
        st.error("Please upload a file smaller than 20 MB.")
        st.stop()

    st.audio(raw)

    if st.button("Analyze recording", type="primary"):
        try:
            with st.spinner("Extracting audio features..."):
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / ("query" + suffix)
                    path.write_bytes(raw)
                    vector = extract_features(path)

                model = bundle["model"]
                prediction = model.predict(vector.reshape(1, -1))[0]

                st.subheader("Predicted dataset category")
                st.info(display_names.get(prediction, str(prediction)))
                st.caption(
                    "This is a category prediction, not a diagnosis or "
                    "a probability that the vehicle has a fault."
                )

                with sf.SoundFile(io.BytesIO(raw)) as audio_file:
                    sr = audio_file.samplerate
                    audio = audio_file.read(
                        frames=sr * 30, dtype="float32", always_2d=True
                    ).mean(axis=1)

                left, right = st.columns(2)
                with left:
                    st.subheader("Waveform")
                    step = max(1, len(audio) // 3000)
                    waveform = pd.DataFrame(
                        {"Amplitude": audio[::step]},
                        index=np.arange(0, len(audio), step) / sr
                    )
                    waveform.index.name = "Time (seconds)"
                    st.line_chart(waveform)

                with right:
                    st.subheader("Frequency spectrum")
                    freq, power = welch(
                        audio, fs=sr, nperseg=min(2048, len(audio))
                    )
                    mask = freq <= min(8000, sr / 2)
                    spectrum = pd.DataFrame(
                        {"Power (dB)": 10 * np.log10(power[mask] + 1e-12)},
                        index=freq[mask]
                    )
                    spectrum.index.name = "Frequency (Hz)"
                    st.line_chart(spectrum)

                query = scaler.transform(vector.reshape(1, -1))[0]
                distances = np.linalg.norm(references - query, axis=1)
                nearest = np.argsort(distances)[:3]

                st.subheader("Closest reference recordings")
                st.caption(
                    "Ranked by standardized feature distance. A similar "
                    "recording can have a different label from the prediction."
                )

                matches = []
                for rank, index in enumerate(nearest, start=1):
                    relative = bundle["reference_paths"][index]
                    label = str(bundle["reference_labels"][index])
                    st.markdown(
                        f"**{rank}. {display_names.get(label, label)}**"
                    )
                    st.caption(Path(relative).name)
                    reference_path = ROOT / relative
                    if reference_path.is_file():
                        st.audio(reference_path.read_bytes())
                    else:
                        st.caption("Reference audio is not installed.")
                    matches.append({
                        "rank": rank,
                        "label": label,
                        "file": relative,
                        "feature_distance": float(distances[index])
                    })

                report = {
                    "filename": uploaded.name,
                    "predicted_dataset_category": str(prediction),
                    "analyzed_seconds": len(audio) / sr,
                    "reference_matches": matches,
                    "limitation": (
                        "Experimental dataset classification; not a "
                        "validated automotive fault diagnosis."
                    )
                }
                st.download_button(
                    "Download analysis report",
                    data=json.dumps(report, indent=2),
                    file_name="autosound_analysis.json",
                    mime="application/json"
                )

        except Exception as error:
            st.error(f"Could not analyze this recording: {error}")

with st.expander("Recorded evaluation and limitations"):
    metrics_path = ROOT / "results/metrics_v5.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.write(
            f"Test recordings: {metrics['test_recordings']} | "
            f"Accuracy: {metrics['accuracy']:.1%} | "
            f"Macro-F1: {metrics['macro_f1']:.3f}"
        )
        st.write(metrics.get("limitation", ""))
    st.write(
        "Dataset provenance and labels have not been independently verified. "
        "File names suggest some recordings may be sound effects, synthetic "
        "audio, or motorcycles. Performance on new vehicles is unknown."
    )

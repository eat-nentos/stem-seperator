import streamlit as st
import os
import tempfile
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
import requests

# ---------- Page Config ----------
st.set_page_config(
    page_title="Stem Separator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Stem Separator")
st.markdown(
    "Upload a song and split it into **vocals**, **drums**, **bass**, "
    "**guitar**, **piano**, and **other**. Powered by HT-Demucs (ONNX)."
)

# ---------- Model Constants ----------
MODEL_URL = (
    "https://huggingface.co/StemSplitio/htdemucs-6s-onnx/"
    "resolve/main/htdemucs_6s.onnx"
)
MODEL_PATH = "/tmp/htdemucs_6s.onnx"
SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")
SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)  # 343,980
N_CHANNELS = 2


# ---------- Download Model (cached) ----------
@st.cache_resource(show_spinner="Downloading model (first time only, ~258 MB)...")
def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return MODEL_PATH


# ---------- Load ONNX Session (cached) ----------
@st.cache_resource(show_spinner="Loading model...")
def load_session():
    path = download_model()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )


# ---------- Overlap-Add Window ----------
def _make_window(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    w[:overlap] = fade
    w[-overlap:] = fade[::-1]
    return w


# ---------- Separation ----------
def separate(mix: np.ndarray, progress_cb=None) -> dict[str, np.ndarray]:
    """Run htdemucs_6s on ``mix`` (shape ``(channels, samples)``)."""
    session = load_session()

    if mix.ndim != 2 or mix.shape[0] not in (1, 2):
        raise ValueError(f"expected (1|2, samples), got {mix.shape}")
    if mix.shape[0] == 1:
        mix = np.repeat(mix, 2, axis=0)

    total = mix.shape[1]
    overlap = N_SAMPLES // 4
    stride = N_SAMPLES - overlap
    n_chunks = max(1, (total + stride - 1) // stride)
    window = _make_window(N_SAMPLES, overlap)

    out = np.zeros((len(SOURCES), N_CHANNELS, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)

    for i in range(n_chunks):
        start = i * stride
        end = min(start + N_SAMPLES, total)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(
                chunk,
                ((0, 0), (0, N_SAMPLES - chunk.shape[1])),
                mode="constant",
            )
        x = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        stems = session.run(["stems"], {"mix": x})[0][0]  # (6, 2, N)
        clen = end - start
        w = window[:clen]
        out[:, :, start:end] += stems[:, :, :clen] * w
        weight[start:end] += w

        if progress_cb:
            progress_cb(
                0.1 + 0.8 * (i + 1) / n_chunks,
                f"Processing chunk {i + 1} of {n_chunks}...",
            )

    # Normalize by accumulated window weight
    weight = np.maximum(weight, 1e-8)
    out /= weight[np.newaxis, np.newaxis, :]

    return {name: out[i] for i, name in enumerate(SOURCES)}


# ---------- UI ----------
uploaded = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "flac", "ogg", "m4a"],
)

if uploaded:
    st.audio(uploaded)

    if st.button("🚀 Separate", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            # Save upload
            in_path = os.path.join(tmp, uploaded.name)
            with open(in_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Read audio
            try:
                audio, sr = sf.read(
                    in_path, dtype="float32", always_2d=True
                )
            except Exception as e:
                st.error(f"Could not read audio file: {e}")
                st.stop()

            if sr != SAMPLE_RATE:
                st.warning(
                    f"Sample rate is {sr} Hz, but the model expects "
                    f"{SAMPLE_RATE} Hz. Results may be degraded."
                )

            audio = audio.T  # (channels, samples)

            # Run separation
            bar = st.progress(0.0, text="Starting...")

            def cb(frac, msg):
                bar.progress(min(frac, 1.0), text=msg)

            try:
                stems = separate(audio, progress_cb=cb)
                bar.progress(1.0, text="Done!")
                st.success("Separation complete! Download your stems below.")

                for name, arr in stems.items():
                    out_path = os.path.join(tmp, f"{name}.wav")
                    sf.write(out_path, arr.T, SAMPLE_RATE)
                    with open(out_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {name}.wav",
                            data=f.read(),
                            file_name=f"{name}.wav",
                            mime="audio/wav",
                            use_container_width=True,
                        )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption(
    "⏱️ On the free tier, expect ~3–8 minutes per 3-minute song. "
    "Only upload audio you have the rights to use."
)

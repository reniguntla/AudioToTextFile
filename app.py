from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st
import whisper
from pydub import AudioSegment

SUPPORTED_TYPES = ["wav", "aiff", "pcm", "mp3", "aac"]
PCM_SAMPLE_RATES = [8000, 16000, 22050, 32000, 44100, 48000]


@st.cache_resource(show_spinner=False)
def load_model() -> whisper.Whisper:
    return whisper.load_model("base")


def to_wav_bytes(uploaded_file, suffix: str, pcm_sample_rate: int) -> bytes:
    if suffix == "pcm":
        raw_data = uploaded_file.read()
        audio = AudioSegment(
            data=raw_data,
            sample_width=2,
            frame_rate=pcm_sample_rate,
            channels=1,
        )
    else:
        audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=suffix)
    return audio.set_channels(1).set_frame_rate(16000).export(format="wav").read()


def transcribe_audio(wav_bytes: bytes) -> str:
    model = load_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.write(wav_bytes)
        temp_path = temp_wav.name

    try:
        result = model.transcribe(temp_path)
        return result.get("text", "").strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Audio Transcript Generator", page_icon="🎙️")
    st.title("🎙️ Audio Transcript Generator")
    st.write(
        "Upload an audio file and generate a transcript. Supported formats: "
        + ", ".join(SUPPORTED_TYPES)
        + "."
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
    )

    pcm_sample_rate = 16000
    if uploaded_file and uploaded_file.name.lower().endswith(".pcm"):
        pcm_sample_rate = st.selectbox(
            "PCM sample rate",
            options=PCM_SAMPLE_RATES,
            index=1,
            help="Choose the sample rate of the raw PCM input.",
        )

    if uploaded_file:
        suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
        st.audio(uploaded_file)
        if st.button("Generate Transcript", type="primary"):
            with st.spinner("Transcribing audio..."):
                try:
                    wav_bytes = to_wav_bytes(uploaded_file, suffix, pcm_sample_rate)
                    transcript = transcribe_audio(wav_bytes)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unable to transcribe this file: {exc}")
                    return

            if transcript:
                st.subheader("Transcript")
                st.write(transcript)
                st.download_button(
                    "Download transcript",
                    transcript,
                    file_name="transcript.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No speech detected in this audio.")


if __name__ == "__main__":
    main()

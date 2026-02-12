# Audio Transcript Generator (Streamlit)

A Streamlit app that generates a transcript from uploaded audio files.

## Supported formats

- `.wav`
- `.aiff`
- `.pcm` (raw 16-bit mono PCM; select sample rate in UI)
- `.mp3`
- `.aac`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app uses OpenAI Whisper (`base` model) for transcription.
- `ffmpeg` must be available on your system path for format conversions.

# Streamlit SLM Chat Application

A Streamlit-based interactive chat application for Small Language Models (SLMs).

## Features

- Model selection from a dropdown (default: **Phi-3**)
- Clear model status indicator in the UI
- Conversational chat interface with chronological history
- Memory of the last **10** question-answer interactions
- Automatic removal of oldest interactions when memory exceeds limit
- Context-aware response generation using recent conversation history
- Streaming response output for better interactivity
- Context/token limit handling to keep prompts within model bounds
- One-click **Clear Conversation** reset button

## Supported Models

- Phi-3 (`microsoft/Phi-3-mini-4k-instruct`) — default
- Gemma (`google/gemma-2-2b-it`) — optional
- Mistral (`mistralai/Mistral-7B-Instruct-v0.3`) — optional

> Note: optional models may require additional local resources or access permissions.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in your terminal (typically `http://localhost:8501`).


## Hugging Face Token (`HF_KEY`)

Set a Hugging Face access token to download gated/private models:

```bash
export HF_KEY=your_huggingface_token
```

You can also set `HF_KEY` in Streamlit secrets (`.streamlit/secrets.toml`):

```toml
HF_KEY = "your_huggingface_token"
```

The app reads `HF_KEY` from Streamlit secrets first, then environment variables.

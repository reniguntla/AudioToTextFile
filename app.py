from __future__ import annotations

import os
from threading import Thread
from typing import Generator

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_OPTIONS = {
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma": "google/gemma-2-2b-it",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
DEFAULT_MODEL = "Phi-3"
MAX_HISTORY_PAIRS = 10
MAX_CONTEXT_TOKENS = 3500
MAX_NEW_TOKENS = 256


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_id: str, hf_token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token,
    )
    return tokenizer, model


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, str]] = []




def get_hf_token() -> str | None:
    secrets_token = st.secrets.get("HF_KEY") if hasattr(st, "secrets") else None
    env_token = os.getenv("HF_KEY")
    token = secrets_token or env_token
    return token.strip() if isinstance(token, str) and token.strip() else None

def build_messages(history: list[dict[str, str]], user_input: str) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and concise assistant.",
        }
    ]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def enforce_context_limit(
    tokenizer,
    history: list[dict[str, str]],
    user_input: str,
) -> torch.Tensor:
    working_history = history.copy()

    while working_history:
        messages = build_messages(working_history, user_input)
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if tokenized.shape[-1] <= MAX_CONTEXT_TOKENS:
            return tokenized
        working_history.pop(0)

    messages = build_messages([], user_input)
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if tokenized.shape[-1] > MAX_CONTEXT_TOKENS:
        tokenized = tokenized[:, -MAX_CONTEXT_TOKENS:]
    return tokenized


def stream_model_response(
    model_name: str,
    history: list[dict[str, str]],
    user_input: str,
    hf_token: str | None,
) -> Generator[str, None, None]:
    model_id = MODEL_OPTIONS[model_name]
    tokenizer, model = load_model_and_tokenizer(model_id, hf_token)
    input_ids = enforce_context_limit(tokenizer, history, user_input).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for chunk in streamer:
        yield chunk


def render_history(history: list[dict[str, str]]) -> None:
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])


def main() -> None:
    st.set_page_config(page_title="SLM Chat Interface", page_icon="💬")
    st.title("💬 Streamlit SLM Chat")
    st.caption("Chat with small language models using short-term conversation memory.")

    init_state()
    hf_token = get_hf_token()

    selected_model = st.selectbox(
        "SLM Selection",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
    )
    st.info(f"Currently Using Model: {selected_model}")
    if hf_token:
        st.caption("Hugging Face access token detected via HF_KEY.")
    else:
        st.warning("HF_KEY is not configured. Private or gated Hugging Face models may fail to load.")

    clear_clicked = st.button("Clear Conversation")
    if clear_clicked:
        st.session_state.chat_history = []
        st.success("Conversation cleared.")

    st.subheader("Conversation Window")
    render_history(st.session_state.chat_history)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Question Input", placeholder="Type your question here")
        submit = st.form_submit_button("Submit", type="primary")

    if submit and user_input.strip():
        question = user_input.strip()
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for piece in stream_model_response(
                    selected_model,
                    st.session_state.chat_history,
                    question,
                    hf_token,
                ):
                    full_response += piece
                    response_placeholder.markdown(full_response + "▌")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Generation failed for {selected_model}: {exc}")
                return

            response_placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"question": question, "answer": full_response.strip()}
        )
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_PAIRS:]
    elif submit:
        st.warning("Please enter a question before submitting.")


if __name__ == "__main__":
    main()

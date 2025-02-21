import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# ----------------------------------------------------------------
# CSS Styling for Chat Layout & Fixed Bottom Input Bar
# ----------------------------------------------------------------
st.markdown(
    """
    <style>
    .chat-container {
        margin-bottom: 80px; /* create space for the bottom input bar */
    }
    .chat-message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        max-width: 70%;
    }
    .chat-message.user {
        background-color: #DCF8C6;
        text-align: right;
        margin-left: auto;
    }
    .chat-message.ai {
        background-color: #F1F0F0;
        text-align: left;
        margin-right: auto;
    }
    .bottom-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        background: white;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Model Selection & Unloading Previous Model
# ----------------------------------------------------------------
chatmodels_list = [
    "Llama-2-7b-chat-hf",
    "deepseek-qwen-1.5B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-3B",
    "Llama-3.2-3B-Instruct"
]

st.title("WhatsApp-Style Chatbot")
MODEL = st.sidebar.selectbox("Select a model:", chatmodels_list)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../models/{MODEL}")
)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Using device: {device}")

# If the model selection changes, clear cached resources so the previous model unloads
if "current_model_name" not in st.session_state or st.session_state.current_model_name != MODEL:
    st.cache_resource.clear()
    st.session_state.current_model_name = MODEL

@st.cache_resource()
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model

st.sidebar.write("Loading model...")
tokenizer, model = load_model(MODEL_PATH)
st.sidebar.success("Model loaded successfully!")

# ----------------------------------------------------------------
# Chat History & Rendering
# ----------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a placeholder to render the chat history dynamically.
chat_placeholder = st.empty()

def render_chat():
    with chat_placeholder.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-message user">{chat["user"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-message ai">{chat["ai"]}</div>',
                unsafe_allow_html=True,
            )
        # If a streaming message exists, show it.
        if "streaming_message" in st.session_state:
            st.markdown(
                f'<div class="chat-message ai">{st.session_state.streaming_message}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# Render existing chat history on app load.
render_chat()

# ----------------------------------------------------------------
# Fixed Bottom Input Bar Using a Form
# ----------------------------------------------------------------
st.markdown('<div class="bottom-input">', unsafe_allow_html=True)
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_input")
    submitted = st.form_submit_button("Send")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Process Input & Stream LLM Response
# ----------------------------------------------------------------
if submitted and user_input:
    # Append the user's message with an empty AI reply (to be filled via streaming).
    st.session_state.chat_history.append({"user": user_input, "ai": ""})
    # Create a temporary key for the streaming response.
    st.session_state.streaming_message = ""
    render_chat()  # Update chat to show the user's message immediately

    # Tokenize the user input and send it to the model.
    inputs = tokenizer(user_input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generated_ids = input_ids
    max_new_tokens = 50  # Adjust as needed

    # Stream the model's output token-by-token.
    for _ in range(max_new_tokens):
        outputs = model.generate(
            generated_ids,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        next_token = outputs[:, -1].unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        # Decode the current sequence and update the streaming message.
        current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        st.session_state.streaming_message = current_text
        render_chat()
        time.sleep(0.1)  # Brief pause for visible streaming effect
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Finalize the AI message in chat history.
    st.session_state.chat_history[-1]["ai"] = st.session_state.streaming_message
    del st.session_state.streaming_message
    render_chat()

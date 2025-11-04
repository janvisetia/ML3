
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import random
# Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Load Pretrained Models

models = {
    "Model 1: Context=5 | Embed=64 | Activation=ReLU | Patience=3": {
        "path": "next_word_model.h5",
        "context_len": 5
    },
    "Model 2: Context=7 | Embed=32 | Activation=Tanh | Patience=3": {
        "path": "model_context7_tanh.h5",
        "context_len": 7
    },
    "Model 3: Context=10 | Embed=64 | Activation=ReLU | Patience=5": {
        "path": "model_context10_relu.h5",
        "context_len": 10
    },
}

# Streamlit Title
st.title("🧠 Next Word Predictor (Shakespeare Dataset)")
st.markdown("Built using an MLP trained on Shakespeare’s text — choose a model, tweak parameters, and generate text.")

# Sidebar Controls
st.sidebar.header("⚙️ Model & Hyperparameters")

model_choice = st.sidebar.selectbox("Select Model:", list(models.keys()))
context_len = models[model_choice]["context_len"]
model_path = models[model_choice]["path"]

# Load the selected model
@st.cache_resource
def load_selected_model(path):
    return tf.keras.models.load_model(path)

model = load_selected_model(model_path)

# Temperature and Randomness
temperature = st.sidebar.slider("Temperature (controls randomness)", 0.1, 2.0, 1.0, 0.1)
num_words = st.sidebar.number_input("Number of words to generate", 1, 100, 10)
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

# Set random seed
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
# Text Input

input_text = st.text_input("Enter your starting text:", "the king said")

# Helper: Temperature Sampling

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate Next Words

def generate_text(seed_text, num_words, temperature, model, context_len):
    text = seed_text.lower().strip()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = token_list[-context_len:]
        if len(token_list) < context_len:
            token_list = [0] * (context_len - len(token_list)) + token_list
        token_list = np.array(token_list).reshape(1, context_len)
        preds = model.predict(token_list, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        next_word = tokenizer.index_word.get(next_index, "")
        if next_word:
            text += " " + next_word
    return text

# Generate Button

if st.button("🚀 Generate"):
    if input_text.strip():
        generated_text = generate_text(input_text, num_words, temperature, model, context_len)
        st.subheader("✨ Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter some seed text first!")

# Footer
st.markdown("---")
st.caption("NLP MLP Text Generator with Streamlit")

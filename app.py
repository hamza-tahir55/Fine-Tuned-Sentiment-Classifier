import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    print("Loading model and tokenizer...")  # Debug statement
    model = AutoModelForSequenceClassification.from_pretrained("hamza-tahir55/IMDB_fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("hamza-tahir55/IMDB_fine_tuned_model")
    print("Model and tokenizer loaded.")  # Debug statement
    return model, tokenizer

# Call the function to load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit UI setup
st.title("IMDB Sentiment Analysis App")

# User input
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        positive_prob = probabilities[0][1].item()
        negative_prob = 1 - positive_prob  # Negative probability is 1 minus positive

        st.write(f"Positive Sentiment Probability: {positive_prob:.2%}")
        st.write(f"Negative Sentiment Probability: {negative_prob:.2%}")
    else:
        st.warning("Please enter text to analyze.")

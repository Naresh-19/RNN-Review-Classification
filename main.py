import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import streamlit as st 

word_index = imdb.get_word_index()
reversed_word_index = {value : key for key,value in word_index.items()}

model = load_model('simple_rnn.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
# Convert each word into its corresponding index using the word_index dictionary.
# If the word is not found, use the index 2 (for unknown words).
# The +3 shift ensures that indices 0, 1, and 2 are reserved for special tokens:
#   0: Padding token (to ensure input length consistency)
#   1: Start token (indicates the beginning of the sequence)
#   2: Unknown token (for words not found in the vocabulary)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
  
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
 
    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]
  
  
#streamlit app

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Entter a Movie review to classofy it as positive or negative")

#User Input

user_input = st.text_area('Movie Review...')

if st.button('Classify'):
    if not user_input.strip():
        st.warning('Please Enter a Movie Review to classify !!', icon="⚠️")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Positivity: {confidence * 100:.0f}%')


  
  

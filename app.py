import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


model = pickle.load(open('hate_speech_detection_model.pkl', 'rb'))
vectorizer = pickle.load(open('hate_speech_detection_vectorizer.pkl', 'rb'))

st.title("Hate Speech Detector")
st.subheader(
    'Provide an amharic text to detect hate speech')

new_text = st.text_input('Text')

if st.button('Detect hate speech'):
    vectorized = vectorizer.transform([new_text]) #
    predicted = model.predict(vectorized)
    if predicted[0] == 'Free':
        st.write(f'Hate speech not detected')
    else:
        st.write(f'Hate speech detected')
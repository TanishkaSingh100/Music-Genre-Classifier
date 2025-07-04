import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import gdown
import os
from skimage.transform import resize
import platform

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "About"])

# Load model and classes
if platform.system() == "Windows":
    model_file = "Trained_model_v2.h5"
else:
    model_file = "/tmp/Trained_model_v2.h5"

file_id = "1PB1reLQmwirfjMVKi6SXri2Q3UOK2xxm"
url = "https://drive.google.com/uc?id=1PB1reLQmwirfjMVKi6SXri2Q3UOK2xxm"

if not os.path.exists(model_file):
    gdown.download(url, model_file, quiet=False)

model = tf.keras.models.load_model(model_file)
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Preprocessing with chunking
def preprocess_file(file_path, duration=4, target_shape=(120, 120), overlap=2):
    import soundfile as sf
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        chunk_len = sr * duration
        overlap_len = sr * overlap
        step = chunk_len - overlap_len

        chunks = []
        for start in range(0, len(audio) - chunk_len + 1, step):
            chunk = audio[start:start + chunk_len]
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr)
            mel_resized = resize(np.expand_dims(mel, axis=-1), target_shape)
            chunks.append(mel_resized)

        return np.array(chunks)

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return []

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <h1 style='text-align: center; color: #87CEFA;'> Music Genre Classifier </h1>
        <p style='text-align: center; color: #6c6c6c;'>Upload a .wav audio file and let the model predict its genre!</p>
        """,
        unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Upload your .wav file")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner('Analyzing...'):
            try:
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.read())

                data = preprocess_file("temp.wav")
                if len(data) == 0:
                    st.stop()

                predictions = model.predict(data)
                total_probs = np.mean(predictions, axis=0)

                predicted_class = classes[np.argmax(total_probs)]
                st.success(f"Predicted Genre: {predicted_class}")

                st.subheader("Confidence for Each Genre")
                fig, ax = plt.subplots()
                ax.bar(classes, total_probs, color="skyblue")
                plt.xticks(rotation=45)
                plt.ylabel("Confidence")
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")

# About Page
elif app_mode == "About":
    st.markdown(
        """
        <h1 style='text-align: center; color: #87CEFA;'> About the Model </h1>
        """,
        unsafe_allow_html=True)
    st.markdown("""
    This Music Genre Classifier is a Deep Learning project developed using TensorFlow, trained on the GTZAN music dataset.

    - Goal: Classify an audio file into one of 10 music genres.
    - Genres: ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    - Model: CNN with Mel Spectrogram features
    - Frameworks: TensorFlow, Librosa, Streamlit
    - Dataset: 1000 audio files across 10 genres

    Developed By: Tanishka Singh

    ---
    """)
    
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Thankyou for checking out the model!</p>", unsafe_allow_html=True)

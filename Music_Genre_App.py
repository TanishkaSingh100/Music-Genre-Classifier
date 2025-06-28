import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import matplotlib.pyplot as plt
import os

# Sidebar Navigation

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to",["Home","About"])

# Load model and classes

model = tf.keras.models.load_model("Trained_model_v2.h5")
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Audio Preprocessing Function

def preprocess_file(audio_file_path, target_shape=(120, 120)):
    data = []
    audio_data, sr = librosa.load(audio_file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2
    chunk_sample = chunk_duration * sr
    overlap_sample = overlap_duration * sr

    num_chunk = int(np.ceil((len(audio_data) - chunk_sample) / (chunk_sample - overlap_sample))) + 1
    for i in range(num_chunk):
        start = i * (chunk_sample - overlap_sample)
        end = start + chunk_sample
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Home Page

if app_mode == "Home":
    #st.title("Music Genre Classifier")
    st.markdown(
    """
    <h1 style='text-align: center; color: #87CEFA;'> Music Genre Classifier </h1>
    <p style='text-align: center; color: #6c6c6c;'>Upload a .wav audio file and let the model predict its genre!</p>
    """,
    unsafe_allow_html=True)
    st.markdown("---")
    
# Upload audio file

    st.subheader("Upload your .wav file")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# Prediction logic
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner('Analyzing...'):
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.read())

        data = preprocess_file("temp.wav")
        predictions = model.predict(data)
        predicted_class = classes[np.argmax(np.sum(predictions, axis=0))]

        st.success(f"Predicted Genre: *{predicted_class}*")

        # Plot confidence
        st.subheader("Confidence for Each Genre")
        total_probs = np.sum(predictions, axis=0)
        fig, ax = plt.subplots()
        ax.bar(classes, total_probs, color="skyblue")
        plt.xticks(rotation=45)
        plt.ylabel("Confidence")
        plt.tight_layout()
        st.pyplot(fig)

# ABOUT PAGE
elif app_mode == "About":
    st.markdown(
    """
    <h1 style='text-align: center; color: #87CEFA;'> About the Model </h1>,
    """,
    unsafe_allow_html=True)
    #st.title(" About the Model")
    st.markdown("""
    This Music Genre Classifier is a Deep Learning project developed using TensorFlow, trained on the GTZAN music dataset.

    - *Goal:* Classify an audio file into one of 10 music genres.
    - *Genres:* ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    - *Model:* CNN with Mel Spectrogram features
    - *Frameworks:* TensorFlow, Librosa, Streamlit
    - *Dataset:* 1000 audio files across 10 genres
    
    
    *Developed By:* Tanishka Singh

    ---
    """)

    # Image
    image_path = "genre_visual.jpg"
    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption="Music Genre Classifier", use_container_width=True)
    
# Footer

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Thankyou for checking out the model!</p>",
    unsafe_allow_html=True
)
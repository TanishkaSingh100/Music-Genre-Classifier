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
import soundfile as sf

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to",["Home","About"])

# Load model and classes
if platform.system() == "Windows":
    model_file = "Trained_model_v2.h5"
else:
    model_file = "/tmp/Trained_model_v2.h5"

file_id = "1PB1reLQmwirfjMVKi6SXri2Q3UOK2xxm"
url ="https://drive.google.com/uc?id=1PB1reLQmwirfjMVKi6SXri2Q3UOK2xxm"

if not os.path.exists(model_file):
    gdown.download(url, model_file, quiet=False)

model = tf.keras.models.load_model(model_file)
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Audio Preprocessing Function
def preprocess_file(audio_file_path, target_shape=(120, 120)):
    data = []
    try:
        audio_data, sr = sf.read(audio_file_path)
        st.write(f"Audio loaded with soundfile: len={len(audio_data)}, sr={sr}")
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            st.write("Stereo converted to mono.")
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        return []

    chunk_duration = 4
    overlap_duration = 2
    chunk_sample = chunk_duration * sr
    overlap_sample = overlap_duration * sr

    try:
        num_chunk = int(np.ceil((len(audio_data) - chunk_sample) / (chunk_sample - overlap_sample))) + 1
        st.write(f"chunk_sample: {chunk_sample}, overlap_sample: {overlap_sample}, num_chunk: {num_chunk}")
    except Exception as e:
        st.error(f"Chunking calculation failed: {e}")
        return []

    for i in range(min(num_chunk, 3)):
        try:
            start = i * (chunk_sample - overlap_sample)
            end = start + chunk_sample
            chunk = audio_data[start:end]
            st.write(f"Chunk {i+1}: length = {len(chunk)}")

            if np.isnan(chunk).any() or np.max(np.abs(chunk)) < 1e-5:
                st.warning(f"Chunk {i+1} skipped: silent or invalid")
                continue

            st.write(f"Generating mel spectrogram for chunk {i+1}...")
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
            st.write(f"Chunk {i+1}: mel_spectrogram shape = {mel_spectrogram.shape}")

            mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            st.write(f"Chunk {i+1}: resized shape = {mel_spectrogram_resized.shape}")

            data.append(mel_spectrogram_resized)
            st.success(f"✅ Chunk {i+1} processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing chunk {i+1}: {e}")

    st.write(f"Final preprocessed data shape: {np.array(data).shape}")
    return np.array(data)

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
                st.info(" Saving file...")
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.read())
                st.info("File Saved...")

                st.info(" Preprocessing audio...")
                data = preprocess_file("temp.wav")
                if len(data) == 0:
                    st.error("No valid audio chunks to process.")
                    st.stop()
                st.info(" Preprocessing complete...")

                st.info("Predicting genre...")
                predictions = model.predict(data)
                st.info("Prediction complete...")

                predicted_class = classes[np.argmax(np.sum(predictions, axis=0))]
                st.success(f"🎧 Predicted Genre: {predicted_class}")

                st.subheader("Confidence for Each Genre")
                total_probs = np.sum(predictions, axis=0)
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

    image_url = "https://raw.githubusercontent.com/TanishkaSingh100/Music-Genre-Classifier/main/genre_visual.jpg"
    st.image(image_url, caption="Music Genre Classifier", use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Thankyou for checking out the model!</p>",
    unsafe_allow_html=True
)
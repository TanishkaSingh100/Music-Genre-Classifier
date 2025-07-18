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
from pydub import AudioSegment
import shutil

# --- check ffmpeg
ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")

if not ffmpeg_path or not ffprobe_path:
    st.error("ffmpeg or ffprobe not found in PATH. Please install FFmpeg and add it to your PATH.")
    st.stop()

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe   = ffprobe_path

# st.write(f"ffmpeg found at: {ffmpeg_path}")
# st.write(f"ffprobe found at: {ffprobe_path}")

# --- Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "About"])

# --- Load model & classes
if platform.system() == "Windows":
    model_file = "Trained_model_v2.h5"
else:
    model_file = "/tmp/Trained_model_v2.h5"

url = "https://drive.google.com/uc?id=1PB1reLQmwirfjMVKi6SXri2Q3UOK2xxm"

if not os.path.exists(model_file):
    st.info("Downloading model…")
    gdown.download(url, model_file, quiet=False)

model = tf.keras.models.load_model(model_file)
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# --- Preprocessing
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

def convert_mp3_to_wav(mp3_path, wav_path="temp.wav"):
    try:
        # st.write(f"Converting {mp3_path} → {wav_path} …")
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Failed to convert mp3 to wav: {e}")
        return None

# --- Home Page
if app_mode == "Home":
    st.markdown("<h1 style='text-align:center;color:#87CEFA;'>🎵 Music Genre Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6c6c6c;'>Upload an audio file (.wav or .mp3) and let the model predict its genre!</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Upload your audio file")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner('Analyzing…'):
            try:
                file_ext = uploaded_file.name.split(".")[-1].lower()

                if file_ext == "wav":
                    with open("temp.wav", "wb") as f:
                        f.write(uploaded_file.read())

                elif file_ext == "mp3":
                    with open("temp.mp3", "wb") as f:
                        f.write(uploaded_file.read())
                    converted_path = convert_mp3_to_wav("temp.mp3")
                    if not converted_path:
                        st.stop()
                else:
                    st.error("Unsupported file format.")
                    st.stop()

                data = preprocess_file("temp.wav")
                if len(data) == 0:
                    st.stop()

                predictions = model.predict(data)
                total_probs = np.mean(predictions, axis=0)

                predicted_class = classes[np.argmax(total_probs)]
                st.success(f"Predicted Genre: *{predicted_class}*")

                st.subheader("Confidence for Each Genre")
                fig, ax = plt.subplots()
                ax.bar(classes, total_probs, color="skyblue")
                plt.xticks(rotation=45)
                plt.ylabel("Confidence")
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# --- About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align:center;color:#87CEFA;'>About the Model</h1>", unsafe_allow_html=True)
    st.markdown("""
    This Music Genre Classifier is a Deep Learning project developed using TensorFlow, trained on the GTZAN music dataset.

    - *Goal:* Classify an audio file into one of 10 music genres.
    - *Genres:* blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
    - *Model:* CNN trained on Mel Spectrograms.
    - *Frameworks:* TensorFlow, Librosa, Streamlit.
    - *Dataset:* 1000 audio files across 10 genres.

    Developed By: *Tanishka Singh*
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:grey;'>✨ Thank you for checking out the model!</p>", unsafe_allow_html=True)
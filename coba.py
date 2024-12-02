import os
import pickle
import streamlit as st

# Mendapatkan path direktori saat ini
current_dir = os.getcwd()
st.write("Current Directory:", current_dir)

# List file yang ada di direktori
files = os.listdir(current_dir)
st.write("Files in Current Directory:", files)

# Membuat path ke file tertentu (misalnya, 'model_terbaik.pkl')
file_path = os.path.join(current_dir, "model_terbaik.pkl")
st.write("Path to File:", file_path)

# Mengecek keberadaan file
if os.path.exists(file_path):
    st.success("File ditemukan COE!")

    # Membuka file pickle dan memuat model
    try:
        with open(file_path, "rb") as file:
            model_terbaik = pickle.load(file)
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Error saat membuka file pickle: {e}")
else:
    st.error("File tidak ditemukan!")

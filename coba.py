import os
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
    st.success("File ditemukan!")
else:
    st.error("File tidak ditemukan!")

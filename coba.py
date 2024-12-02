from joblib import load

try:
    model_terbaik = load("model_terbaik.pkl")
    print("Model berhasil dimuat:", model_terbaik)
except Exception as e:
    print(f"Error saat membuka file joblib: {e}")

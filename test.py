import pickle

# Buka file lama
with open("model_terbaik.pkl", "rb") as file:
    model = pickle.load(file)

# Simpan ulang file
with open("model_elm_terbaik.pkl", "wb") as file:
    pickle.dump(model, file)
    print("berhasil")

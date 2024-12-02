import sys
import numpy as np
import pandas as pd
import streamlit as st
from hpelm import ELM
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

st.markdown(
    "<h2 style='text-align: center;'><i>Extreme Learning Machine</i> Untuk Memprediksi Curah Hujan Dalam Penentuan Jadwal Tanam Padi</h2><br><br><br>", unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Main Menu", ['Dataset', 'Preprocessing', 'Modelling', 'Prediction'], default_index=3)

# DATASET --------------------------------------------------------------
data = pd.read_excel('data.xlsx', parse_dates=['Tanggal'])
dataset = pd.read_excel('dataset.xlsx')
X = dataset.drop(columns=['Target']).values
y = dataset['Target'].values

# MODELLING ------------------------------------------------------------
# Fungsi untuk pengujian berbagai rasio latih dan uji serta hidden neuron
def run_experiment(dataset, hidden_neurons_range, train_test_ratios):
    X = dataset.drop(columns=['Lag_7'])
    y = dataset['Lag_7']
    results = []
    for ratio in train_test_ratios:
        train_size = int(len(X) * ratio)
        X_train, X_test = X[:train_size].values, X[train_size:].values  # Convert to numpy arrays
        y_train, y_test = y[:train_size].values, y[train_size:].values 
        
        for hidden_neurons in hidden_neurons_range:
            # 4. Membuat Model ELM
            elm = ELM(X_train.shape[1], 1)
            elm.add_neurons(hidden_neurons, 'sigm')
            elm.train(X_train, y_train, 'r')

            # 5. Prediksi
            y_pred_test = elm.predict(X_test)
            
            # 6. Evaluasi dengan MAPE, MAE, dan MSE
            error_mape = safe_mape(y_test, y_pred_test)  # Menggunakan safe_mape
            error_mae = mean_absolute_error(y_test, y_pred_test)
            error_mse = mean_squared_error(y_test, y_pred_test)
            
            # Simpan hasilnya
            results.append({
                'Train/Test Ratio': ratio,
                'Hidden Neurons': hidden_neurons,
                'MAPE': error_mape,  # Simpan sebagai desimal
                'MAE': error_mae,
                'MSE': error_mse
            })
    
    return pd.DataFrame(results)

# Fungsi untuk menghitung MAPE dengan aman (menghindari pembagian dengan nol)
def safe_mape(y_true, y_pred):
    mask = y_true != 0  # Hanya menghitung MAPE ketika nilai y_true tidak nol
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)  # Kembali ke format desimal

current_dir = os.getcwd() # Mendapatkan path direktori saat ini
files = os.listdir(current_dir) # List file yang ada di direktori
file_path = os.path.join(current_dir, "model_terbaik.pkl")

try:
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    st.write("Model berhasil dibuka!")
except Exception as e:
    st.write(f"Error saat membuka model: {e}")

if (selected == 'Dataset'):
    st.info("Data curah hujan harian diperoleh dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). Kabupaten Bangkalan tidak memiliki stasiun pengamatan cuaca, sehingga data curah hujan yang diolah dari hasil pengamatan stasiun pengamatan cuaca terdekat, yaitu Stasiun Meteorologi Perak I Surabaya.")
    st.markdown("[Badan Meteorologi, Klimatologi, dan Geofisika (BMKG)](https://www.bmkg.go.id/)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Data Curah Hujan Harian')
        data

    with col2:
        st.subheader('Informasi Data')
        st.write("Data yang diolah merupakan data curah hujan harian dalam kurun waktu Januari 2020 – Juli 2024")
        st.info(f"Total = {data.shape[0]} data")
        st.write("Terdapat :")
        st.warning(f"1. Jumlah nilai 8888 (Tidak diukur) : {(data == 8888).sum().sum()} data")
        st.warning(f"2. Jumlah data kosong (None)        : {data.isna().sum().sum()} data")

if (selected == 'Preprocessing'):
    st.info("""
    Adapun tahapan - tahapan yang akan dilakukan pada persiapan data ini adalah :
    1. Data Imputation
    2. Data Tranformation
    3. Lag Feature
    4. Normalisasi Data
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Imputation", "Data Tranformation", "Normalisasi Data", "Lag Feature", "Dataset"])
    # Imputasi data
    with tab1:
        st.warning('Data Imputation')
        st.write("Jumlah Missing Values dalam Setiap Kolom : ", data.isnull().sum())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data

        with col2:
            st.subheader('Data Sesudah')
            data['CH'] = data['CH'].fillna(0)
            data
        
        missing_values = data.isnull().sum()
        st.write("Jumlah Missing Values dalam Setiap Kolom Setelah Data Imputation : ", missing_values)

    # Tranformasi data
    with tab2:
        st.warning('Data Transformation')
        st.write(f'Jumlah data dengan nilai 8888 : {(data['CH'] == 8888).sum()}')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data

        with col2:
            st.subheader('Data Sesudah')
            data['CH'] = data['CH'].replace(8888, 0)
            data

        st.write(f'Jumlah data dengan nilai 8888 : {(data['CH'] == 8888).sum()}')

    # Normalisasi data
    with tab3:
        st.warning('Jenis normalisasi data yang digunakan adalah Min-Max Scaler')
        st.write(f"Nilai curah hujan maksimum : {data["CH"].max()}")
        st.write(f"Nilai curah hujan minimum : {data["CH"].min()}")
        
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['CH']])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data['CH'] 

        with col2:
            st.subheader('Data Sesudah')
            data_scaled
        
    with tab4:
        st.warning('Lag Feature')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Fitur')
            X 

        with col2:
            st.subheader('Target')
            y

    with tab5:
        st.warning('Dataset') 
        st.write("Berikut adalah data yang telah melewati tahap preprocessing dan akan digunakan pada tahap modelling")       
        dataset

if (selected == 'Modelling'):

    tab1, tab2 = st.tabs(["Hasil Pengujian", "Uji Coba"])
    with tab1:
        # Menampilkan parameter terbaik
        st.info(f"Parameter terbaik ditemukan pada:")
        st.write(model_terbaik)

    with tab2:
        # Input for minimum and maximum hidden neurons
        hidden_neurons_min = st.slider('Minimum hidden neurons', min_value=1, max_value=20, value=1)
        hidden_neurons_max = st.slider('Maximum hidden neurons', min_value=1, max_value=20, value=10)

        # Input for train-test ratios
        train_test_ratios = st.multiselect(
            'Select train-test ratios',
            options=[0.7, 0.8, 0.9],
            default=[0.7, 0.8, 0.9]  # Default selected ratios
        )

        # When the "Run Experiment" button is clicked
        if st.button('Run Experiment'):
            # Generate hidden neurons range based on user input
            hidden_neurons_range = range(hidden_neurons_min, hidden_neurons_max + 1)

            # Run the experiment
            results = run_experiment(dataset, hidden_neurons_range, train_test_ratios)

            # Find the best result based on the smallest MAPE
            best_result = results.loc[results['MAPE'].idxmin()]  # Get the minimum MAPE row

            # Display the best result
            st.write("Best parameters found:")
            st.write(f"Train/Test Ratio: {best_result['Train/Test Ratio']}")
            st.write(f"Hidden Neurons: {best_result['Hidden Neurons']}")
            st.write(f"MAPE: {best_result['MAPE']}")  # Display MAPE as decimal
            st.write(f"MAE: {best_result['MAE']}")
            st.write(f"MSE: {best_result['MSE']}")

            # Visualize the results for MAPE, MAE, and MSE
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # Create a figure with 3 subplots

            # Plot MAPE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[0].plot(subset['Hidden Neurons'], subset['MAPE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[0].set_xlabel('Hidden Neurons')
            axes[0].set_ylabel('MAPE')
            axes[0].set_title('MAPE vs Hidden Neurons')
            axes[0].legend()

            # Plot MAE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[1].plot(subset['Hidden Neurons'], subset['MAE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[1].set_xlabel('Hidden Neurons')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('MAE vs Hidden Neurons')
            axes[1].legend()

            # Plot MSE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[2].plot(subset['Hidden Neurons'], subset['MSE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[2].set_xlabel('Hidden Neurons')
            axes[2].set_ylabel('MSE')
            axes[2].set_title('MSE vs Hidden Neurons')
            axes[2].legend()

            plt.tight_layout()  # Adjust layout for better spacing

            # Display the figure in Streamlit
            st.pyplot(fig)
            
if (selected == 'Prediction'):

    hidden_neurons_best = int(model_terbaik['Hidden Neurons'])
    best_train_test_ratio = model_terbaik['Train/Test Ratio']

    train_size = int(len(X) * best_train_test_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Membuat Model ELM dengan parameter terbaik
    elm = ELM(X.shape[1], 1)
    elm.add_neurons(hidden_neurons_best, 'sigm')
    elm.train(X_train, y_train)

    # Prediksi 365 hari ke depan
    data['CH'] = data['CH'].fillna(0)
    data['CH'] = data['CH'].replace(8888, 0)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['CH']])

    tab1, tab2 = st.tabs(["Jadwal Tanam Padi", "Prediksi Curah Hujan"])
    with tab1:
        predictions = []
        current_input = data_scaled[-7:].flatten()  # Mengambil data terakhir untuk prediksi
        for _ in range(365):
            # Prediksi
            pred = elm.predict(current_input.reshape(1, -1))
            predictions.append(pred[0][0])
            
            # Update input dengan memasukkan prediksi terbaru
            current_input = np.roll(current_input, -1)  # Geser input
            current_input[-1] = pred  # Masukkan prediksi ke input

        # Denormalisasi data
        predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Membuat DataFrame dari hasil prediksi
        future_dates = pd.date_range(start=data['Tanggal'].max() + pd.Timedelta(days=1), periods=365)
        predicted_df = pd.DataFrame({
            'Tanggal': future_dates,
            'CH Prediksi': predictions_rescaled.flatten()
        })

        # Mengelompokkan berdasarkan bulan untuk menentukan jadwal tanam padi
        predicted_df['Bulan'] = predicted_df['Tanggal'].dt.month

        # Dapatkan bulan dari tanggal terakhir pada dataset asli
        last_month = data['Tanggal'].max().month

        # Menggeser bulan sehingga dimulai dari bulan terakhir
        predicted_df['Bulan'] = (predicted_df['Bulan'] + (last_month - 1)) % 12 + 1
        monthly_summary = predicted_df.groupby('Bulan')['CH Prediksi'].sum().reset_index()

        # Tampilkan hasil ringkasan bulanan
        st.write("Ringkasan Prediksi Bulanan:")
        st.write(monthly_summary)

        fig, ax = plt.subplots(figsize=(10, 5))  # Menggunakan plt.subplots untuk mendapatkan `fig` dan `ax`
        ax.plot(monthly_summary['Bulan'], monthly_summary['CH Prediksi'], marker='o', color='skyblue', linestyle='-')

        ax.set_title('Total Prediksi Curah Hujan per Bulan')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Total Curah Hujan (mm)')

        # Mengatur label x-axis dengan nama bulan dan rotasi
        ax.set_xticks(range(len(monthly_summary['Bulan'])))
        ax.set_xticklabels(['Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun'], rotation=45)
        ax.grid(axis='y')
        st.pyplot(fig)

    with tab2:
        days_to_predict = st.number_input("Masukkan jumlah hari untuk diprediksi (contoh: 30):", min_value=1, max_value=365, value=30, step=1)
        if st.button("Prediksi"):
            predictions = []
            current_input = data_scaled[-7:].flatten()  # Mengambil data terakhir untuk prediksi
            for _ in range(int(days_to_predict)):
                # Prediksi
                pred = elm.predict(current_input.reshape(1, -1))
                predictions.append(pred[0][0])
                
                # Update input dengan memasukkan prediksi terbaru
                current_input = np.roll(current_input, -1)  # Geser input
                current_input[-1] = pred  # Masukkan prediksi ke input

            # Denormalisasi data
            predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Membuat DataFrame dari hasil prediksi
            future_dates = pd.date_range(start=data['Tanggal'].max() + pd.Timedelta(days=1), periods=days_to_predict)
            predicted_df = pd.DataFrame({
                'Tanggal': future_dates,
                'CH Prediksi': predictions_rescaled.flatten()
            })

            # Menampilkan hasil
            st.subheader(f"Hasil Prediksi Curah Hujan untuk {days_to_predict} Hari")
            st.dataframe(predicted_df)

            # Menampilkan grafik prediksi
            st.line_chart(predicted_df.set_index('Tanggal')['CH Prediksi'])

            # Menampilkan hasil untuk hari terakhir
            last_prediction = predictions_rescaled[-1][0]
            if last_prediction > 0 :
                st.success(f"Prediksi curah hujan pada hari ke-{days_to_predict}: {last_prediction:.2f} mm")
            else:
                st.error(f"Tidak turun hujan pada hari ke-{days_to_predict}")


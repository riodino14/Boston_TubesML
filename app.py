# app.py
import numpy as np
import streamlit as st
import pickle
from adaboost_model import AdaBoostR2  # agar pickle bisa mengenali model custom

# === Load model dan scaler ===
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("adaboost_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Gagal memuat model atau scaler: {e}")
    st.stop()

# === Konfigurasi halaman Streamlit ===
st.set_page_config(page_title="Prediksi Harga Rumah Boston", layout="wide")
st.title("🏠 Prediksi Harga Rumah Boston (MEDV) dengan AdaBoostR2")
st.markdown("""
Masukkan nilai fitur-fitur berikut untuk memprediksi **harga median rumah** di Boston dalam ribuan USD.  
Fitur-fitur diambil dari *Boston Housing Dataset* (UCI ML Repository).
""")

# === Informasi fitur ===
feature_info = {
    'CRIM': ("Tingkat kejahatan per kapita", 3.6),
    'ZN': ("% tanah untuk rumah > 25.000 sq.ft", 11.4),
    'INDUS': ("% area bisnis non-retail", 11.1),
    'CHAS': ("1 jika dekat Sungai Charles, 0 jika tidak", 0.0),
    'NOX': ("Konsentrasi NOx (ppm)", 0.55),
    'RM': ("Rata-rata jumlah kamar per rumah", 6.28),
    'AGE': ("% rumah dibangun sebelum 1940", 68.6),
    'DIS': ("Jarak ke pusat kerja (km)", 3.79),
    'RAD': ("Indeks akses ke jalan raya radial", 9.5),
    'TAX': ("Pajak properti per $10.000", 408.2),
    'PTRATIO': ("Rasio siswa-guru", 18.5),
    'B': ("1000(Bk - 0.63)², Bk = % orang kulit hitam", 356.7),
    'LSTAT': ("% populasi berstatus ekonomi rendah", 12.6)
}

# === Form input fitur ===
cols = st.columns(2)
user_vals = []

for i, (feat, (desc, default)) in enumerate(feature_info.items()):
    with cols[i % 2]:
        val = st.number_input(
            label=f"{feat}",
            value=float(default),
            step=0.01,
            format="%.5f",
            help=desc
        )
        user_vals.append(val)

# === Prediksi ===
if st.button("🎯 Prediksi Harga"):
    if all(v == 0 for v in user_vals):
        st.warning("⚠️ Harap isi setidaknya satu fitur sebelum melakukan prediksi.")
    else:
        try:
            x_scaled = scaler.transform([user_vals])
            prediction = model.predict(x_scaled)[0]
            st.success(f"💰 Prediksi MEDV: **${prediction:.2f}k**")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {e}")

# === Footer ===
st.markdown("---")
st.caption("Model: AdaBoostR2 + DecisionTree buatan sendiri | Data: Boston Housing (UCI ML)")

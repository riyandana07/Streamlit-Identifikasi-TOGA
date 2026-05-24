import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Tanaman TOGA",
    page_icon="🌿",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background: linear-gradient(
        135deg,
        #1b4332 0%,
        #2d6a4f 40%,
        #40916c 100%
    );
    background-attachment: fixed;
}
            
/* Judul */
.main-title {
    font-size: 45px;
    font-weight: bold;
    color: #dcfce7;
    text-align: center;
    margin-bottom: 10px;
}

/* Subjudul */
.subtitle {
    text-align: center;
    color: #f0fdf4;
    font-size: 20px;
    margin-bottom: 30px;
}

/* Card */
.custom-card {
    background-color: rgba(255,255,255,0.88);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

/* Hasil prediksi */
.result-box {
    background-color: #dcfce7;
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
    color: #14532d;
    font-weight: bold;
    font-size: 17px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #14532d 0%,
        #166534 100%
    );
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Upload text jadi hitam */
[data-testid="stFileUploader"] {
    color: white;
}

/* Label uploader */
.upload-text {
    color: black;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* List tanaman */
.plant-list {
    background-color: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
    line-height: 1.8;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = YOLO("best (2).pt")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌿 Sistem Deteksi TOGA")

st.sidebar.markdown("""
Aplikasi ini digunakan untuk mendeteksi daun tanaman obat keluarga (TOGA) menggunakan model YOLOv8 berbasis Computer Vision.
""")

st.sidebar.markdown("## 📚 Dataset Tanaman")

st.sidebar.markdown("""
<div class="plant-list">

🌱 Belimbing Wuluh<br>
🌱 Jambu Biji<br>
🌱 Jeruk Nipis<br>
🌱 Kemangi<br>
🌱 Lidah Buaya<br>
🌱 Nangka<br>
🌱 Pandan<br>
🌱 Pepaya<br>
🌱 Seledri<br>
🌱 Sirih

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 🤖 Model
YOLOv8s

### 🖼️ Total Kelas
10 Jenis Tanaman
""")

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">🌿 Deteksi Tanaman TOGA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Sistem Deteksi Daun Tanaman Obat Keluarga Menggunakan YOLOv8</div>',
    unsafe_allow_html=True
)

# =========================
# TEXT UPLOAD
# =========================
st.markdown(
    '<div class="upload-text">📤 Upload Gambar Daun Tanaman</div>',
    unsafe_allow_html=True
)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROSES DETEKSI
# =========================
if uploaded_file is not None:

    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")

    # Convert numpy
    img_array = np.array(image)

    # Prediksi YOLO
    results = model(img_array)

    # Gambar hasil
    annotated_image = results[0].plot()

    # Layout 2 kolom
    col1, col2 = st.columns(2)

    # =========================
    # GAMBAR ASLI
    # =========================
    with col1:

        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True
        )

        st.subheader("📷 Gambar Asli")

        st.image(
            image,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # HASIL DETEKSI
    # =========================
    with col2:

        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True
        )

        st.subheader("✅ Hasil Deteksi")

        st.image(
            annotated_image,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # DETAIL PREDIKSI
    # =========================
    st.markdown(
        '<div class="custom-card">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Detail Prediksi")

    boxes = results[0].boxes

    if len(boxes) > 0:

        for box in boxes:

            conf = float(box.conf[0])

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            st.markdown(
                f"""
                <div class="result-box">
                🌱 Tanaman: {class_name}<br>
                🎯 Confidence: {conf:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(conf)

    else:
        st.warning("❌ Tanaman tidak terdeteksi")

    st.markdown('</div>', unsafe_allow_html=True)
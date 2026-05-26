import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Identifikasi Tanaman Obat Keluarga (TOGA)",
    page_icon="🌿",
    layout="centered"
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
        #2d6a4f 50%,
        #40916c 100%
    );
    min-height: 100vh;
}

/* Menghindari bug layar putih di HP */
html, body, [class*="css"] {
    max-width: 100%;
    overflow-x: hidden;
}

/* Container utama */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1000px;
}

/* Judul */
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #dcfce7;
    text-align: center;
    margin-bottom: 10px;
}

/* Subjudul */
.subtitle {
    text-align: center;
    color: #f0fdf4;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Hasil prediksi */
.result-box {
    background-color: #dcfce7;
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
    color: #14532d;
    font-weight: bold;
    font-size: 16px;
}

/* Informasi tanaman */
.info-box {
    background-color: rgba(255,255,255,0.92);
    padding: 18px;
    border-radius: 18px;
    margin-top: 12px;
    margin-bottom: 15px;
    color: #14532d;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    line-height: 1.7;
    font-size: 15px;
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

/* Upload text */
.upload-text {
    color: white;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
    margin-top: 5px;
    text-align: center;
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

/* Responsive HP */
@media screen and (max-width: 768px) {

    .main-title {
        font-size: 26px !important;
    }

    .subtitle {
        font-size: 15px !important;
    }

    .result-box {
        font-size: 14px !important;
    }

    .upload-text {
        font-size: 18px !important;
    }

    .info-box {
        font-size: 14px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = YOLO("best (1).pt")

# =========================
# INFORMASI TANAMAN
# =========================
plant_info = {

    "Belimbing Wuluh":
    "Membantu meningkatkan daya tahan tubuh dan membantu menjaga tekanan darah.",

    "Jambu Biji":
    "Mengandung vitamin C tinggi dan membantu menjaga sistem pencernaan.",

    "Jeruk Nipis":
    "Membantu meningkatkan daya tahan tubuh dan membantu meredakan flu serta batuk.",

    "Kemangi":
    "Membantu menjaga kesehatan pencernaan dan mengandung antioksidan alami.",

    "Lidah Buaya":
    "Membantu menjaga kesehatan kulit dan rambut serta mengandung antioksidan.",

    "Nangka":
    "Membantu mempercepat penyembuhan luka dan membantu menjaga kesehatan kulit.",

    "Pandan":
    "Membantu memberikan efek relaksasi dan sering digunakan sebagai aromaterapi alami.",

    "Pepaya":
    "Membantu melancarkan pencernaan dan meningkatkan nafsu makan.",

    "Seledri":
    "Membantu menjaga tekanan darah dan kesehatan tubuh.",

    "Sirih":
    "Memiliki sifat antiseptik alami dan membantu menjaga kesehatan mulut."
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌿 Sistem Identifikasi TOGA")

st.sidebar.markdown("""
Aplikasi berbasis web ini digunakan untuk mengidentifikasi jenis tanaman obat keluarga (TOGA) menggunakan model YOLOv8s berbasis Computer Vision.
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
    '<div class="main-title">🌿 Sistem Identifikasi Tanaman Obat Keluarga (TOGA)</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Implementasi YOLOv8s untuk Identifikasi Tanaman Obat Keluarga Berbasis Web</div>',
    unsafe_allow_html=True
)
# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Daun Tanaman",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROSES IDENTIFIKASI
# =========================
if uploaded_file is not None:

    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")

    # Convert ke numpy
    img_array = np.array(image)

    # Prediksi
    results = model(img_array)

    # Gambar hasil
    annotated_image = results[0].plot()

    # =========================
    # GAMBAR ASLI
    # =========================
    st.subheader("📷 Gambar Asli")

    st.image(
        image,
        use_container_width=True
    )

    # =========================
    # HASIL IDENTIFIKASI
    # =========================
    st.subheader("✅ Hasil Identifikasi")

    st.image(
        annotated_image,
        use_container_width=True
    )

    # =========================
    # DETAIL PREDIKSI
    # =========================
    st.subheader("📊 Detail Prediksi")

    boxes = results[0].boxes

    if len(boxes) > 0:

        for box in boxes:

            # Confidence menjadi persen
            conf = float(box.conf[0]) * 100

            # Class ID
            cls_id = int(box.cls[0])

            # Nama class
            class_name = model.names[cls_id]

            # Hasil prediksi
            st.markdown(
                f"""
                <div class="result-box">
                🌱 Tanaman: {class_name}<br>
                🎯 Confidence: {conf:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            # Progress bar
            st.progress(conf / 100)

            # =========================
            # INFORMASI TANAMAN
            # =========================
            info = plant_info.get(
                class_name,
                "Informasi tanaman tidak tersedia."
            )

            st.markdown(
                f"""
                <div class="info-box">
                🌿 <b>Manfaat Tanaman {class_name}</b><br><br>
                {info}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.warning("❌ Tanaman tidak teridentifikasi")

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    """
    <center>
    🌿 Sistem Identifikasi Tanaman TOGA Berbasis YOLOv8s <br>
    Dibuat untuk Penelitian Skripsi
    </center>
    """,
    unsafe_allow_html=True
)

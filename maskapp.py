import cv2
import numpy as np
import streamlit as st

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Segmentasi HSV Interaktif",
    layout="wide"
)

st.title("Aplikasi Segmentasi Citra Berbasis Thresholding HSV")
st.write("Unggah gambar dan sesuaikan rentang ambang batas HSV pada panel samping untuk membuat binary mask secara real-time.")

# Sidebar Pengaturan
st.sidebar.header("Pengaturan Threshold HSV")

# Widget Expander untuk Panduan Warna Hue
with st.sidebar.expander("Panduan Diagram Hue (0-179)", expanded=False):
    st.image("Hue OpenCV.png", caption="Grid Nilai Hue OpenCV (0-179)", width="stretch")

st.sidebar.subheader("Rentang Nilai HSV")

# Menggunakan Range Slider (Tuple nilai awal mengubah widget menjadi 2 titik geser)
h_min, h_max = st.sidebar.slider("Rentang Hue (H)", 0, 179, (111, 124))
s_min, s_max = st.sidebar.slider("Rentang Saturation (S)", 0, 255, (131, 255))
v_min, v_max = st.sidebar.slider("Rentang Value (V)", 0, 255, (20, 255))

# Menyusun array NumPy batas bawah dan atas
lower_lesion = np.array([h_min, s_min, v_min], dtype=np.uint8)
upper_lesion = np.array([h_max, s_max, v_max], dtype=np.uint8)

# Pengolahan Citra
uploaded_file = st.file_uploader("Pilih Berkas Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    lesion_mask = cv2.inRange(img_hsv, lower_lesion, upper_lesion)
    segmented_result = cv2.bitwise_and(img_rgb, img_rgb, mask=lesion_mask)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Citra Asli (RGB)")
        st.image(img_rgb, width="stretch")
    with col2:
        st.subheader("2. Binary Mask")
        st.image(lesion_mask, width="stretch", channels="GRAY")
    with col3:
        st.subheader("3. Hasil Segmentasi")
        st.image(segmented_result, width="stretch")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk menampilkan hasil segmentasi.")

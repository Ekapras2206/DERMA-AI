import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import cv2
import tempfile
import time

# === CONFIGURATION ===
MODEL_PATH = "model/derma_ai_final.pt"
IMG_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]

st.set_page_config(page_title="DERMA-AI (PyTorch)", layout="centered")

# === LOAD MODEL ===
@st.cache_resource
def load_model(path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# === PREPROCESS ===
def preprocess_pil(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# === PREDICT ===
def get_prediction(img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASS_NAMES[idx.item()], conf.item()

# === HEADER ===
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺 (PyTorch)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload atau ambil gambar kulitmu untuk klasifikasi lesi.</p>", unsafe_allow_html=True)

# === PILIH MODE INPUT ===
mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera", "Live Scan"])

img = None

# === MODE 1: UPLOAD FILE ===
if mode == "Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar kulit (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar yang kamu upload", use_container_width=True)

# === MODE 2: CAMERA SNAPSHOT ===
elif mode == "Ambil dari Kamera":
    camera_img = st.camera_input("Ambil foto dari kamera")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        st.image(img, caption="Foto hasil kamera", use_container_width=True)

# === MODE 3: LIVE SCAN ===
elif mode == "Live Scan":
    st.warning("⚠️ Fitur live scan membutuhkan akses webcam (real-time). Tekan 'Mulai Scan' di bawah.")
    start_scan = st.button("📷 Mulai Scan")

    if start_scan:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        scan_button = st.button("📸 Scan Sekarang")
        stop_button = st.button("❌ Berhenti")

        scanned_img = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Kamera tidak terbaca.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            if scan_button:
                scanned_img = Image.fromarray(frame)
                img = scanned_img
                st.success("✅ Gambar berhasil di-scan!")
                break

            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()

# === HASIL PREDIKSI ===
if img is not None:
    img_tensor = preprocess_pil(img)
    label, conf = get_prediction(img_tensor)

    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align: center;'>Prediksi: <b>{label.upper()}</b></h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center;'>Confidence Score: <b>{conf*100:.2f}%</b></p>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px; color: gray;'>Model dilatih menggunakan dataset HAM10000 — hanya untuk tujuan edukasi.</p>", unsafe_allow_html=True)

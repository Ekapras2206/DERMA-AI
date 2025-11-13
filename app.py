import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2

# ==========================
# 🔧 CONFIGURATION
# ==========================
MODEL_PATH = "model/derma_ai_final.pt"
IMG_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]

# Mapping label → nama lengkap
LABEL_MAP = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Lesions"
}

st.set_page_config(page_title="DERMA-AI", layout="centered")

# ==========================
# 🔍 LOAD MODEL
# ==========================
@st.cache_resource
def load_model(path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ==========================
# 🧠 PREPROCESS FUNCTION
# ==========================
def preprocess_pil(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# ==========================
# 🧴 SKIN FILTER (OpenCV)
# ==========================
def is_skin_image(pil_img, threshold=0.05):
    """Cek apakah gambar mengandung area kulit cukup besar"""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rentang warna kulit umum dalam HSV
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    return skin_ratio > threshold, skin_ratio

# ==========================
# 📊 PREDICTION FUNCTION
# ==========================
def get_prediction(img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASS_NAMES[idx.item()], conf.item()

# ==========================
# 🩺 HEADER
# ==========================
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload atau ambil gambar kulitmu untuk klasifikasi lesi.</p>", unsafe_allow_html=True)

# ==========================
# 🖼️ MODE INPUT
# ==========================
mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera"])
img = None

# === MODE 1: UPLOAD FILE ===
if mode == "Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar kulit (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar yang kamu upload")

# === MODE 2: CAMERA SNAPSHOT ===
elif mode == "Ambil dari Kamera":
    camera_img = st.camera_input("📷 Ambil foto dari kamera")
    if camera_img is not None:
        try:
            img = Image.open(camera_img).convert("RGB")
            st.image(img, caption="Foto hasil kamera")
        except Exception as e:
            st.error(f"❌ Gagal membaca gambar dari kamera: {e}")
    else:
        st.info("Silakan ambil foto terlebih dahulu.")

# ==========================
# 📈 HASIL PREDIKSI (Upload / Kamera)
# ==========================
if img is not None:
    # Deteksi apakah gambar mengandung kulit
    is_skin, ratio = is_skin_image(img)

    if not is_skin:
        st.warning("⚠️ Gambar ini tidak terdeteksi sebagai kulit. "
                   "Silakan upload foto bagian kulit manusia yang jelas.")
    else:
        img_tensor = preprocess_pil(img)
        short_label, conf = get_prediction(img_tensor)
        full_label = LABEL_MAP.get(short_label, short_label)  # ambil nama panjang

        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align: center;'>Prediction: <b>{full_label}</b></h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center;'>Confidence Score: <b>{conf*100:.2f}%</b></p>",
            unsafe_allow_html=True
        )

# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Model dilatih menggunakan dataset HAM10000 — hanya untuk tujuan edukasi.</p>",
    unsafe_allow_html=True
)




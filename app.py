import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time

# ==========================
# 🔧 CONFIGURATION
# ==========================
MODEL_PATH = "model/derma_ai_final.pt"
IMG_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]

st.set_page_config(page_title="DERMA-AI (PyTorch)", layout="centered")

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
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺 (PyTorch)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload atau ambil gambar kulitmu untuk klasifikasi lesi.</p>", unsafe_allow_html=True)

# ==========================
# 🖼️ MODE INPUT
# ==========================
mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera", "Live Scan (beta)"])
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

# === MODE 3: LIVE SCAN (SIMULATED LOOP) ===
elif mode == "Live Scan":
    st.warning("🟢 Mode Live Scan (simulasi) — cocok untuk kamera HP.")

    if "scan_active" not in st.session_state:
        st.session_state.scan_active = False

    start_btn = st.button("▶️ Mulai Scan")
    stop_btn = st.button("⏹️ Berhenti")

    if start_btn:
        st.session_state.scan_active = True
    if stop_btn:
        st.session_state.scan_active = False

    if st.session_state.scan_active:
        live_img = st.camera_input("Ambil frame untuk dianalisis")
        if live_img is not None:
            try:
                img = Image.open(live_img).convert("RGB")
                st.image(img, caption="Frame terbaru")

                # Prediksi langsung
                img_tensor = preprocess_pil(img)
                label, conf = get_prediction(img_tensor)
                st.markdown(
                    f"<h3 style='text-align:center;'>Prediksi: {label.upper()} — {conf*100:.2f}%</h3>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Error saat memproses frame: {e}")
        else:
            st.info("📸 Arahkan kamera ke kulit dan ambil gambar untuk melihat prediksi.")

# ==========================
# 📈 HASIL PREDIKSI (untuk Upload / Kamera)
# ==========================
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

# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Model dilatih menggunakan dataset HAM10000 — hanya untuk tujuan edukasi.</p>",
    unsafe_allow_html=True
)


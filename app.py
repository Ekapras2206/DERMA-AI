import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

# ==========================
# 🔧 CONFIGURATION
# ==========================
MODEL_PATH = "model/derma_ai_best.pt"
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
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
# 🔥 GRAD-CAM LOCALIZATION
# ==========================
def gradcam_on_image(model, img_tensor):
    model.eval()
    img_tensor = img_tensor.to(device)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    fwd.remove()
    bwd.remove()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

# ==========================
# 🎯 CAM → Bounding Box
# ==========================
def get_bounding_box_from_cam(cam, orig_img):
    h, w = orig_img.size[1], orig_img.size[0]
    cam_resized = cv2.resize(cam, (w, h))

    thresh = cam_resized > 0.4
    coords = np.column_stack(np.where(thresh))

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (x_min, y_min, x_max, y_max)

# ==========================
# 🩺 UI HEADER
# ==========================
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload atau ambil gambar kulitmu untuk klasifikasi lesi.</p>", unsafe_allow_html=True)

# ==========================
# 🖼️ MODE INPUT
# ==========================
mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera"])
img = None

if mode == "Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar kulit (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar yang kamu upload")

elif mode == "Ambil dari Kamera":
    camera_img = st.camera_input("📷 Ambil foto dari kamera")
    if camera_img is not None:
        img = Image.open(camera_img).convert("RGB")
        st.image(img, caption="Foto hasil kamera")

# ==========================
# 📈 HASIL PREDIKSI + BBOX
# ==========================
if img is not None:
    is_skin, ratio = is_skin_image(img)

    if not is_skin:
        st.warning("⚠️ Gambar ini tidak terdeteksi sebagai kulit.")
    else:
        img_tensor = preprocess_pil(img)
        short_label, conf = get_prediction(img_tensor)
        full_label = LABEL_MAP.get(short_label, short_label)

        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align: center;'>Prediction: <b>{full_label}</b></h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center;'>Confidence Score: <b>{conf*100:.2f}%</b></p>",
            unsafe_allow_html=True
        )

        # ===== GRAD-CAM =====
        cam = gradcam_on_image(model, img_tensor)
        bbox = get_bounding_box_from_cam(cam, img)

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            img_np = np.array(img)
            img_bbox = img_np.copy()
            cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 3)

            st.markdown("### 📌 Deteksi Area Lesi (Bounding Box)")
            st.image(img_bbox, use_column_width=True)
        else:
            st.info("Tidak ditemukan area lesi yang jelas dari Grad-CAM.")

# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Model dilatih menggunakan dataset HAM10000 — hanya untuk tujuan edukasi.</p>",
    unsafe_allow_html=True
)

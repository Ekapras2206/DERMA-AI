import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

# ==========================
# 🔧 CONFIG
# ==========================
MODEL_PATH = "model/derma_ai_best.pt"
IMG_SIZE = 224

# Tambahkan normal & nonskin
CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "nv",
    "mel", "vasc", "normal", "nonskin"
]

LABEL_MAP = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Lesions",
    "normal": "Healthy Skin",
    "nonskin": "Non-Skin Image"
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
# 🧠 PREPROCESS
# ==========================
def preprocess_pil(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)

# ==========================
# 🧴 SKIN FILTER
# ==========================
def is_skin_image(pil_img, threshold=0.05):
    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    return skin_ratio > threshold, skin_ratio

# ==========================
# 📊 PREDICTION
# ==========================
def get_prediction(img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASS_NAMES[idx.item()], conf.item()

# ==========================
# 🔥 GRAD-CAM
# ==========================
def gradcam_on_image(model, img_tensor):
    model.eval()
    img_tensor = img_tensor.to(device)

    act = []
    grad = []

    def fwd_hook(m, i, o):
        act.append(o)

    def bwd_hook(m, gi, go):
        grad.append(go[0])

    target_layer = model.features[-1]
    f = target_layer.register_forward_hook(fwd_hook)
    b = target_layer.register_backward_hook(bwd_hook)

    out = model(img_tensor)
    cls = out.argmax()

    model.zero_grad()
    out[0, cls].backward()

    f.remove()
    b.remove()

    # 🟩 FIX: detach sebelum numpy()
    grad_val = grad[0].detach().cpu().numpy()[0]
    act_val = act[0].detach().cpu().numpy()[0]

    weights = np.mean(grad_val, axis=(1, 2))
    cam = np.zeros(act_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act_val[i]

    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-9
    return cam


# ==========================
# 🎯 BOUNDING BOX (improved)
# ==========================
def get_bounding_box_from_cam(cam, orig_img):
    W, H = orig_img.size
    heatmap = cv2.resize(cam, (W, H))

    mask = heatmap > 0.3
    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Perbesar box supaya lebih mencakup semua lesi
    pad = int(0.1 * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_max = min(H - 1, y_max + pad)

    return x_min, y_min, x_max, y_max

# ==========================
# 🩺 UI
# ==========================
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺</h1>", unsafe_allow_html=True)

mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera"])
img = None

if mode == "Upload Gambar":
    file = st.file_uploader("Upload JPG/PNG", ["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img)

else:
    cam = st.camera_input("Ambil foto")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img)

# ==========================
# 🚀 PROCESS
# ==========================
if img is not None:

    # Cek apakah gambar kulit
    is_skin, ratio = is_skin_image(img)

    img_tensor = preprocess_pil(img)
    label, conf = get_prediction(img_tensor)
    full_label = LABEL_MAP[label]

    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align:center;'>Prediction: <b>{full_label}</b></h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align:center;'>Confidence: <b>{conf*100:.2f}%</b></p>",
        unsafe_allow_html=True
    )

    # Bila normal / nonskin → tidak perlu GradCAM
    if label in ["normal", "nonskin"]:
        st.info("Tidak menampilkan bounding box karena gambar termasuk **kulit normal** atau **bukan kulit**.")
    else:
        cam = gradcam_on_image(model, img_tensor)
        bbox = get_bounding_box_from_cam(cam, img)

        if bbox:
            x1, y1, x2, y2 = bbox
            img_np = np.array(img)
            img_box = img_np.copy()
            cv2.rectangle(img_box, (x1, y1), (x2, y2), (255, 0, 0), 3)

            st.markdown("### 📌 Deteksi Area Lesi")
            st.image(img_box, use_column_width=True)
        else:
            st.info("Lesi tidak terdeteksi jelas oleh Grad-CAM.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px;'>Model ini hanya untuk edukasi.</p>",
    unsafe_allow_html=True
)


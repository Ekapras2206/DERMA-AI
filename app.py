import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch.nn.functional as F

# ==========================
# 🔧 CONFIG
# ==========================
MODEL_PATH = "model/derma_ai_best.pt"
IMG_SIZE = 224

# ✅ UPDATE: Menambahkan 'scabies' sesuai urutan abjad folder
CLASS_NAMES = [
    "akiec",    # 0
    "bcc",      # 1
    "bkl",      # 2
    "df",       # 3
    "mel",      # 4
    "nonskin",  # 5
    "normal",   # 6
    "nv",       # 7
    "scabies",  # 8  <-- BARU
    "vasc"      # 9
]

# ✅ UPDATE: Menambahkan nama lengkap untuk Scabies
LABEL_MAP = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nonskin": "Non-Skin Image",
    "normal": "Healthy Skin",
    "nv": "Melanocytic Nevi",
    "scabies": "Scabies Infestation", # <-- BARU
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
    # Output layer disesuaikan dengan panjang CLASS_NAMES baru (10 kelas)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    # Load state dict
    # Pastikan file .pt ini adalah hasil training terbaru dengan 10 kelas!
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
except RuntimeError as e:
    st.error(f"❌ Gagal memuat model! Kemungkinan model yang dipakai masih versi lama (9 kelas) tapi kode meminta 10 kelas. \nDetail: {e}")
    st.stop()

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
# 🎯 BOUNDING BOX
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

    pad = int(0.1 * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_max = min(H - 1, y_max + pad)

    return x_min, y_min, x_max, y_max

# ==========================
# 🎨 OVERLAY BBOX
# ==========================
def draw_box_on_pil(img_pil: Image.Image, bbox: tuple):
    x1, y1, x2, y2 = bbox
    draw = ImageDraw.Draw(img_pil)
    # Warna merah tebal
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
    return img_pil

# ==========================
# 🩺 UI
# ==========================
st.markdown("<h1 style='text-align: center;'>DERMA-AI 🩺</h1>", unsafe_allow_html=True)

mode = st.radio("Pilih cara input gambar:", ["Upload Gambar", "Ambil dari Kamera"])
img_slot = st.empty() 
original_img = None 

if mode == "Upload Gambar":
    file = st.file_uploader("Upload JPG/PNG", ["jpg", "jpeg", "png"])
    if file:
        original_img = Image.open(file).convert("RGB")
        img_slot.image(original_img)

else:
    cam = st.camera_input("Ambil foto")
    if cam:
        original_img = Image.open(cam).convert("RGB")
        img_slot.image(original_img)

# ==========================
# 🚀 PROCESS
# ==========================
if original_img is not None:
    processed_img = original_img.copy()

    # Cek apakah gambar kulit
    is_skin, ratio = is_skin_image(processed_img)

    img_tensor = preprocess_pil(processed_img)
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

    # Logic untuk menampilkan Bbox
    # Scabies termasuk yang PERLU bbox, jadi tidak dimasukkan ke list pengecualian
    if label not in ["normal", "nonskin"]:
        cam = gradcam_on_image(model, img_tensor)
        bbox = get_bounding_box_from_cam(cam, processed_img)

        if bbox:
            processed_img = draw_box_on_pil(processed_img, bbox)
            img_slot.image(processed_img)
            st.success("Area lesi terdeteksi.")
        else:
            st.warning("Lesi tidak terdeteksi jelas oleh Grad-CAM.")
            img_slot.image(processed_img)

    else:
        st.info("Tidak menampilkan penandaan karena gambar termasuk **kulit normal** atau **bukan kulit**.")
        img_slot.image(processed_img)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px;'>Model ini hanya untuk edukasi.</p>",
    unsafe_allow_html=True
)
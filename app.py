import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import tempfile

from video import sample_video_frames

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Off-road Navigation Intelligence", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stSidebar"] { min-width: 320px; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    # ---- Terrain Classifier (YOUR TRAINED MODEL) ----
    clf = models.resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 4)
    if os.path.exists("terrain_classifier.pth"):
        clf.load_state_dict(torch.load("terrain_classifier.pth", map_location="cpu"))
    clf.eval()

    # ---- UNet (Pretrained, No Training) ----
    import segmentation_models_pytorch as smp
    unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        activation=None
    )
    unet.eval()

    return clf, unet

clf_model, unet_model = load_models()

# ================= TRANSFORMS =================
clf_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

seg_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASSES = ["Easy", "Moderate", "Rough", "Very Rough"]

# ================= CORE LOGIC =================
def classify_terrain(img):
    x = clf_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(clf_model(x), dim=1)[0]
        idx = torch.argmax(probs).item()
    return CLASSES[idx], float(probs[idx] * 100)

def unet_segment(img):
    x = seg_tf(img).unsqueeze(0)
    with torch.no_grad():
        pred = unet_model(x).squeeze().cpu().numpy()
    mask = (pred > 0).astype(np.uint8)  # binary mask
    return mask

def split_zones(mask):
    h, w = mask.shape
    return (
        mask[:, :w//3],
        mask[:, w//3:2*w//3],
        mask[:, 2*w//3:]
    )

def free_ratio(zone):
    return np.sum(zone == 1) / zone.size

def navigation_decision(mask, terrain_label):
    left, front, right = split_zones(mask)
    lf, ff, rf = free_ratio(left), free_ratio(front), free_ratio(right)

    if terrain_label == "Very Rough":
        decision = "STOP / AVOID"
    elif ff < 0.3:
        decision = "TURN LEFT" if lf > rf else "TURN RIGHT"
    else:
        decision = "GO STRAIGHT"

    return lf, ff, rf, decision

# ================= UI =================
st.title("🚙 Off-road Navigation Intelligence")
st.caption("UNet-based drivable region estimation + trained terrain classifier")

with st.sidebar:
    st.header("Controls")

    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    run_img = st.button("Run Image Analysis", use_container_width=True)

    st.markdown("---")

    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    run_vid = st.button("Run Video Analysis", use_container_width=True)

# ================= IMAGE PIPELINE =================
if img_file and run_img:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    terrain, conf = classify_terrain(img)
    mask = unet_segment(img)
    lf, ff, rf, decision = navigation_decision(mask, terrain)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Terrain Difficulty", terrain, f"{conf:.2f}%")
        st.metric("Navigation Decision", decision)

    with col2:
        st.write("**Zone Free Ratios**")
        st.write(f"Left:  {lf:.2f}")
        st.write(f"Front: {ff:.2f}")
        st.write(f"Right: {rf:.2f}")

    st.subheader("Drivable Region (UNet)")
    st.image(mask * 255, caption="White = Drivable | Black = Blocked", use_container_width=True)

# ================= VIDEO PIPELINE =================
if vid_file and run_vid:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(vid_file.read())
    video_path = tfile.name

    frames = sample_video_frames(video_path, frame_skip=10, max_frames=5)
    decisions = []

    st.subheader("Sampled Video Frames Analysis")

    for i, frame in enumerate(frames):
        terrain, _ = classify_terrain(frame)
        mask = unet_segment(frame)
        _, _, _, decision = navigation_decision(mask, terrain)
        decisions.append(decision)

        st.markdown(f"**Frame {i+1} → {decision}**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(frame, use_container_width=True)
        with c2:
            st.image(mask * 255, use_container_width=True)
        st.divider()

    # Final aggregated decision
    final_decision = max(set(decisions), key=decisions.count)
    st.success(f"Final Video-Level Decision: {final_decision}")

st.divider()
st.caption(
    "Final decisions are generated using spatial analysis of UNet outputs "
    "combined with a trained terrain classification model."
)

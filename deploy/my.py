import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os
from pyzbar import zbar_library
from pyzbar.pyzbar import decode
import utils.virt_env as venv


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Halal & Barcode Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-header {font-size: 2.8rem; font-weight: 700; text-align: center; color: #1E3A8A; margin-bottom: .5rem;}
.subtitle {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
.success-badge {background:#d4edda;color:#155724;padding:.5rem 1rem;border-radius:12px;font-weight:bold;display:inline-block;}
.warning-badge {background:#fff3cd;color:#856404;padding:.5rem 1rem;border-radius:12px;font-weight:bold;display:inline-block;}
.stButton>button {background:#10B981;color:white;border-radius:8px;padding:.5rem 1rem;font-weight:bold;}
.footer {text-align:center;margin-top:3rem;color:#888;font-size:.9rem;}
.upload-box {border:2px dashed #10B981;border-radius:12px;padding:2rem;text-align:center;background:#f9fdfa;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load YOLO Models
# -----------------------------
@st.cache_resource
def load_models():
    with st.spinner("Loading Halal Model..."):
        halal_model = YOLO("halal_logo_detector.pt")
    with st.spinner("Loading Barcode Model..."):
        barcode_model = YOLO("barcode_detector.pt")
    return halal_model, barcode_model

halal_model, barcode_model = load_models()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("**Halal & Barcode Detector**")
    st.markdown("Detect halal logos and barcodes on product packaging.")
    halal_conf = st.slider("Halal Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    barcode_conf = st.slider("Barcode Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    st.markdown("#### Tips")
    st.markdown("- Clear, well-lit photos\n- Ensure logo/barcode is visible\n- Avoid reflections")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<h1 class='main-header'>Halal & Barcode Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture a product image to detect halal logos and barcodes.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"], label_visibility="collapsed")
with col2:
    camera_image = st.camera_input("Or Take a Photo", label_visibility="collapsed")

image_source = camera_image if camera_image else uploaded_file

# -----------------------------
# Processing
# -----------------------------
if image_source is not None:
    pil_image = Image.open(image_source).convert("RGB")
    st.image(pil_image, caption="Input Image", use_container_width=True)

    # Halal Detection
    with st.spinner("Detecting Halal Logos..."):
        halal_results = halal_model.predict(pil_image, conf=halal_conf, verbose=False)[0]

    # Barcode Detection
    with st.spinner("Detecting Barcodes..."):
        barcode_results = barcode_model.predict(pil_image, conf=barcode_conf, verbose=False)[0]

    # Combine Annotated Outputs (barcode overlay last)
    annotated_pil = halal_results.plot(line_width=3, font_size=1.2, pil=True)
    annotated_pil = barcode_results.plot(line_width=2, font_size=1.0, pil=True)

    st.image(annotated_pil, caption="Detection Results", use_container_width=True)

    # -----------------------------
    # Halal Results
    # -----------------------------
    if halal_results.boxes and len(halal_results.boxes)>0:
        st.markdown("<div class='success-badge'>HALAL CERTIFIED</div>", unsafe_allow_html=True)
        st.success(f"**{len(halal_results.boxes)} halal logo(s) detected!**")
        st.markdown("#### Detected Logos:")
        for i, box in enumerate(halal_results.boxes):
            cls_id = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            label = halal_model.names[cls_id]
            st.markdown(f"**{i+1}. {label.title()}** – Confidence: {conf:.2f}")
    else:
        st.markdown("<div class='warning-badge'>NO HALAL LOGO FOUND</div>", unsafe_allow_html=True)

    # -----------------------------
    # Manual ZBar DLL Load + Barcode Decoding
    # -----------------------------
    # dll_path = r"C:\Users\My pc\Downloads\zbar-x64\libzbar-64.dll"  # 👈 adjust this path

    try:
        if os.path.exists(dll_path):
            zbar_library.load(dll_path)
            decoded_barcodes = decode(pil_image)
        else:
            st.warning(f"⚠️ libzbar DLL not found at: {dll_path}\nDownload from https://github.com/NaturalHistoryMuseum/pyzbar/releases")
            decoded_barcodes = []
    except Exception as e:
        st.error(f"❌ Failed to load libzbar or decode barcodes: {e}")
        decoded_barcodes = []

    if barcode_results.boxes and len(barcode_results.boxes)>0:
        st.markdown("### 📦 Barcode Detected")
        st.info(f"{len(barcode_results.boxes)} barcode(s) found!")

        if decoded_barcodes:
            st.markdown("#### Decoded Barcodes:")
            for i, obj in enumerate(decoded_barcodes):
                data = obj.data.decode("utf-8")
                typ = obj.type
                st.markdown(f"**{i+1}. Type:** `{typ}`  |  **Data:** `{data}`")
        else:
            st.warning("Barcode detected, but could not decode. Try a clearer image.")

    # -----------------------------
    # Download annotated result
    # -----------------------------
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Annotated Result",
        data=buf.getvalue(),
        file_name="halal_barcode_result.png",
        mime="image/png"
    )

else:
    st.markdown("""
    <div class='upload-box'>
        <h3>Ready to scan!</h3>
        <p>Upload or capture an image to detect halal logos and barcodes.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class='footer'>
Built with ❤️ using YOLOv8 & Streamlit | Halal + Barcode Verification
</div>
""", unsafe_allow_html=True)

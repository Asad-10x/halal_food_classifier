import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
from pyzbar import zbar_library as zbarlib
from pyzbar.pyzbar import decode, ZBarSymbol


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
    uploaded_file = st.file_uploader("Upload Logo Image", type=["jpg","jpeg","png"], label_visibility="collapsed")
with col2:
    camera_image = st.camera_input("Or Take a Photo", label_visibility="collapsed")
image_source_a = uploaded_file or camera_image
# -----------------------------
# Processing
# -----------------------------
if image_source_a is not None:
    pil_image = Image.open(image_source_a).convert("RGB")
    st.image(pil_image, caption="Input Image", width='stretch')

    # Halal Detection
    with st.spinner("Detecting Halal Logos..."):
        halal_results = halal_model.predict(pil_image, conf=halal_conf, verbose=True)[0]
    annotated_pil = halal_results.plot(line_width=3, font_size=1.2, pil=True)

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
#  Barcode Decoding
# -----------------------------

col3 = st.columns([1])[0]
with col3:
    barcode_upload = st.file_uploader("Upload Barcode Image", type=["jpg","jpeg","png"], label_visibility="collapsed")

image_source_b = barcode_upload or None
annotated_pil_b = None
if image_source_b is not None:
    pil_b = Image.open(image_source_b).convert("RGB")
    # st.image(pil_b, caption="Barcode Image", width='stretch') # shows uploaded image (redundant)

    decoded_barcodes = []
    try:
        decoded_barcodes = decode(pil_b)
    except Exception as e:
        st.warning(f"Could not decode barcodes with pyzbar: {e}")
        decoded_barcodes = []

    st.markdown("### Decoding Barcodes with pyzbar")
    st.write(f"Found {len(decoded_barcodes)} decoded object(s)")

    # Draw pyzbar annotations (rectangles, polygons, text)
    annotated_decode = pil_b.copy()
    draw = ImageDraw.Draw(annotated_decode)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()

    if decoded_barcodes:
        for d in decoded_barcodes:
            # rect: left, top, width, height
            l, t, w, h = d.rect.left, d.rect.top, d.rect.width, d.rect.height
            draw.rectangle(((l, t), (l + w, t + h)), outline=(0, 0, 255), width=3)
            if getattr(d, 'polygon', None):
                try:
                    pts = [(p.x, p.y) for p in d.polygon]
                    draw.polygon(pts, outline=(0, 255, 0))
                except Exception:
                    pass
            try:
                text = d.data.decode('utf-8')
            except Exception:
                text = str(d.data)
            draw.text((l, t + h), text, fill=(255, 0, 0), font=font)

        st.image(annotated_decode, caption="Decoded Barcodes (pyzbar)", width='stretch')

        st.markdown("#### Decoded Barcodes:")
        for i, obj in enumerate(decoded_barcodes):
            data = obj.data.decode("utf-8")
            typ = obj.type
            st.markdown(f"**{i+1}. Type:** `{typ}`  |  **Data:** `{data}`")
    else:
        st.warning("No barcodes decoded by pyzbar. Try a clearer image or ensure libzbar is available.")
    result_img = annotated_pil_b # add an indent if uncomment above block

    if result_img is not None:
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
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

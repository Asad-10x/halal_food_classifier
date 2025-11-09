import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2 # ✅ Import for OpenCV
import os

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
    """Loads and caches the YOLO models."""
    try:
        with st.spinner("Loading Halal Model..."):
            halal_model = YOLO("halal_logo_detector.pt")
        with st.spinner("Loading Barcode Model..."):
            barcode_model = YOLO("barcode_detector.pt")
        return halal_model, barcode_model
    except FileNotFoundError as e:
        st.error(f"Model file not found. Ensure models are in the same directory: {e}")
        return None, None

halal_model, barcode_model = load_models()

# Check if models loaded successfully
if halal_model is None or barcode_model is None:
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
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
    
    # FIX: Replaced use_container_width=True with width="stretch"
    st.image(pil_image, caption="Input Image", width="stretch") 

    # Halal Detection
    with st.spinner("Detecting Halal Logos..."):
        halal_results = halal_model.predict(pil_image, conf=halal_conf, verbose=False)[0]

    # Barcode Detection
    with st.spinner("Detecting Barcodes..."):
        # We run the YOLO barcode detector just to draw the bounding box
        barcode_results = barcode_model.predict(pil_image, conf=barcode_conf, verbose=False)[0]

    # Combine Annotated Outputs (barcode overlay last for visibility)
    annotated_pil = halal_results.plot(line_width=3, font_size=1.2, pil=True)
    # Re-run detection on the annotated image to combine visualizations
    annotated_pil = barcode_results.plot(img=annotated_pil, line_width=2, font_size=1.0, pil=True)

    # FIX: Replaced use_container_width=True with width="stretch"
    st.image(annotated_pil, caption="Detection Results (YOLO Output)", width="stretch") 

    # -----------------------------
    # Halal Results
    # -----------------------------
    st.markdown("## ✨ Halal Status")
    if halal_results.boxes and len(halal_results.boxes) > 0:
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
        st.warning("No certified Halal logo was detected at the set confidence threshold. Check ingredients to confirm status.")

    # -----------------------------
    # OpenCV Barcode Decoding (FINAL ROBUST FIX)
    # -----------------------------
    st.markdown("## 📦 Barcode Decoder")
    decoded_barcodes = []
    try:
        # Convert PIL Image (RGB) to OpenCV format (BGR numpy array)
        img_np_rgb = np.array(pil_image)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Initialize Barcode Detector 
        barcode_detector = cv2.barcode.BarcodeDetector()
        
        # Use a single call and capture the output in a variable
        result = barcode_detector.detectAndDecode(img_np_bgr)
        
        # Check the length of the returned tuple (handle 2, 3, or 4 item returns)
        if isinstance(result, tuple) and len(result) >= 3:
            
            # Use conditional assignment to ensure we use [] if the element is None
            decoded_info = result[1] if result[1] is not None else []
            decoded_type = result[2] if result[2] is not None else []
            
            # Process the results. zip([]) is safe and prevents 'NoneType' error.
            for data, type_name in zip(decoded_info, decoded_type):
                if data and type_name:
                    decoded_barcodes.append({
                        "type": type_name,
                        "data": data
                    })
        # If length is < 3, decoding failed, and decoded_barcodes remains empty.

    except Exception as e:
        # Catch any remaining OpenCV errors
        st.error(f"❌ OpenCV Barcode decoding error: {e}") 

    if barcode_results.boxes and len(barcode_results.boxes) > 0:
        st.info(f"YOLO detected {len(barcode_results.boxes)} barcode region(s).")

        if decoded_barcodes:
            st.markdown("#### Decoded Barcodes:")
            for i, bc in enumerate(decoded_barcodes):
                st.markdown(f"**{i+1}. Type:** `{bc['type']}`  |  **Data:** `{bc['data']}`")
        else:
            st.warning("Barcode region detected by YOLO, but decoding failed with OpenCV. Try a clearer image.")
    else:
        st.warning("No barcodes found by the YOLO detector.")

    # -----------------------------
    # Download annotated result
    # -----------------------------
    st.markdown("---")
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
Built with ❤️ using YOLOv8, OpenCV & Streamlit | Halal + Barcode Verification
</div>
""", unsafe_allow_html=True)
import streamlit as st                                       # pyright: ignore[reportMissingImports]
from ultralytics import YOLO                                 # type: ignore
from PIL import Image, ImageDraw, ImageFont                  # pyright: ignore[reportMissingImports]
import numpy as np                                           # pyright: ignore[reportMissingImports]
import io, os
from pyzbar import zbar_library as zbarlib                   # pyright: ignore[reportMissingImports]
from pyzbar.pyzbar import decode, ZBarSymbol                 # pyright: ignore[reportMissingImports]



# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Computer Vision: Halal Logo & Barcode Detection",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
* {
    transition: all 0.3s ease;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #be774d 0%, #92130c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 10px rgba(190, 119, 77, 0.15);
}

.subtitle {
    font-size: 1.3rem;
    color: #be5738;
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}

.success-badge {
    background: linear-gradient(135deg, #be774d 0%, #92130c 100%);
    color: #ffecd6;
    padding: 0.75rem 1.25rem;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
    box-shadow: 0 8px 16px rgba(146, 19, 12, 0.25);
    border: 2px solid #be774d;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}

.success-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(146, 19, 12, 0.35);
    background: linear-gradient(135deg, #92130c 0%, #be774d 100%);
}

.warning-badge {
    background: linear-gradient(135deg, #ffc8f6 0%, #be5738 100%);
    color: #92130c;
    padding: 0.75rem 1.25rem;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
    box-shadow: 0 8px 16px rgba(255, 200, 246, 0.3);
    border: 2px solid #ffc8f6;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}

.warning-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(255, 200, 246, 0.4);
    background: linear-gradient(135deg, #be5738 0%, #ffc8f6 100%);
}

.stButton>button {
    background: linear-gradient(135deg, #be774d 0%, #be5738 100%) !important;
    color: #ffecd6 !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    border: 2px solid #92130c !important;
    box-shadow: 0 6px 16px rgba(190, 119, 77, 0.3) !important;
    text-transform: uppercase !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.4px !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 24px rgba(146, 19, 12, 0.4) !important;
    background: linear-gradient(135deg, #92130c 0%, #be5738 100%) !important;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    color: #be5738;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.2px;
}

.upload-box {
    border: 2px dashed #be774d;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(255, 236, 214, 0.8) 0%, rgba(255, 200, 246, 0.1) 100%);
    transition: all 0.4s ease;
}

.upload-box:hover {
    border-color: #92130c;
    box-shadow: 0 12px 32px rgba(190, 119, 77, 0.2);
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(255, 236, 214, 0.95) 0%, rgba(255, 200, 246, 0.15) 100%);
}

.upload-box h3 {
    color: #92130c;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.upload-box p {
    color: #be5738;
    font-size: 1rem;
    font-weight: 500;
}
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
    st.markdown("**Halal Logo & Barcode Detector**")
    st.markdown("Detect halal logos and barcodes on product packaging.")
    halal_conf = st.slider("Halal Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    barcode_conf = st.slider("Barcode Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    st.markdown("#### Tips")
    st.markdown("- Clear, well-lit photos\n- Ensure logo/barcode or both are visible\n- Avoid reflections")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<h1 class='main-header'>Halal Logo & Barcode Detector</h1>", unsafe_allow_html=True)
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
    image_one = Image.open(image_source_a).convert("RGB")
    st.image(image_one, caption="Input Image", width='stretch')

    # Halal Detection
    with st.spinner("Detecting Halal Logos..."):
        halal_results = halal_model.predict(image_one, conf=halal_conf, verbose=True)[0]
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
# first detect barcode in uploaded image, if none found, show section to upload barcode image
decoded_barcodes = None
result_img = None
def detect_barcodes_pyzbar(pil_image):
    output = []
    try:
        output = decode(pil_image, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128, ZBarSymbol.QRCODE])
    except Exception as e:
        st.warning(f"Could not decode barcodes with pyzbar: {e}")
        output = []
    return output

def annotate_barcodes(pil_image, decoded_barcodes):
    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()

    if decoded_barcodes is not None:
        for d in decoded_barcodes:
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
        return annotated_image

if uploaded_file is not None:
    decoded_barcodes = detect_barcodes_pyzbar(image_one)
    # print(decoded_barcodes)
if decoded_barcodes:
    result_img = annotate_barcodes(image_one, decoded_barcodes)
    st.markdown("### Decoding Barcodes...")
    st.write(f"Found {len(decoded_barcodes)} decoded object(s)")
    st.image(result_img, caption="Decoded Barcodes (pyzbar)", width='stretch')
    st.markdown("#### Decoded Barcodes:")
    for i, obj in enumerate(decoded_barcodes):
        data = obj.data.decode("utf-8")
        typ = obj.type
        st.markdown(f"##### **{i+1}. Type:** `{typ}`  |  **Data:** `{data}`")
else:
    st.markdown("### No barcodes detected in the image.")
    st.markdown(' You can upload a separate barcode image for decoding.')
    col3 = st.columns([1])[0]
    with col3:
        barcode_upload = st.file_uploader("Upload Barcode Image", type=["jpg","jpeg","png"], label_visibility="collapsed")

    image_source_b = barcode_upload or None
    annotated_pil_b = None
    if image_source_b is not None:
        pil_b = Image.open(image_source_b).convert("RGB")
        decoded_barcodes = detect_barcodes_pyzbar(pil_b)
        annotated_pil_b = annotate_barcodes(pil_b, decoded_barcodes)
        st.markdown("### Decoding Barcodes...")
        st.write(f"Found {len(decoded_barcodes)} decoded object(s)")
        st.image(annotated_pil_b, caption="Decoded Barcodes (pyzbar)", width='stretch')
        st.markdown("#### Decoded Barcodes:")
        for i, obj in enumerate(decoded_barcodes):
            data = obj.data.decode("utf-8")
            typ = obj.type
            st.markdown(f"**{i+1}. Type:** `{typ}`  |  **Data:** `{data}`")
    else:
        st.warning("No barcodes detected in the image. Try a clearer image.")
    result_img = annotated_pil_b 

if result_img is not None:
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Annotated Result",
        data=buf.getvalue(),
        file_name="halal_barcode_result.png",
        mime="image/png"
    )
# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class='footer'>
Built with ❤️ using YOLOv8 & Streamlit | Halal Logo & Barcode Verification
</div>
""", unsafe_allow_html=True)

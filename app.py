import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import tempfile
import os
import cv2

# PAGE CONFIG
st.set_page_config(
    page_title="AI Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

# SESSION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# CUSTOM CSS
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    color: #2c3e50;
}
.footer {
    text-align: center;
    font-size: 13px;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# LOSS & METRICS (for loading model)
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_f = tf.cast(y_pred_f > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_unet.keras",
        custom_objects={
            "dice_coef": dice_coef,
            "iou_metric": iou_metric,
            "bce_dice_loss": bce_dice_loss
        }
    )

model = load_model()

# HEADER
st.markdown('<div class="main-title">🛢️ AI-Based Oil Spill Detection System</div>', unsafe_allow_html=True)
st.markdown("### Real-time Satellite Image Analysis using Deep Learning")

# IMAGE UPLOAD
st.markdown('<div class="section-title">📤 Upload Satellite Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a satellite image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# MAIN PIPELINE
if uploaded_file is not None:

    with st.spinner("🔍 Analyzing image for oil spill..."):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load & resize image EXACTLY like training
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((256, 256))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0

        # Prediction
        pred = model.predict(np.expand_dims(image_np, axis=0))[0].squeeze()

        # LOWER threshold for oil detection
        pred_binary = (pred > 0.3).astype(np.uint8)

        # Morphological cleaning
        kernel = np.ones((3,3), np.uint8)
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)

        oil_percentage = (pred_binary.sum() / pred_binary.size) * 100

        # LOWER decision threshold
        decision = "Oil Spill Detected" if oil_percentage > 0.2 else "No Oil Spill Detected"

        # ADD TO HISTORY
        st.session_state.history.append({
            "timestamp": timestamp,
            "decision": decision,
            "oil_percentage": oil_percentage
        })

    # RESULTS
    st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

    st.write(f"**Decision:** {decision}")
    st.write(f"**Oil Spill Percentage:** {oil_percentage:.2f}%")
    st.write(f"**Timestamp:** {timestamp}")

    # VISUALIZATION
    st.markdown('<div class="section-title">🖼️ Visual Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_resized, caption="Original Image", use_container_width=True)

    with col2:
        st.image(pred_binary * 255, caption="Predicted Mask", use_container_width=True)

    with col3:
        fig, ax = plt.subplots()
        ax.imshow(image_resized)
        ax.imshow(pred_binary, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)
        
    # PROBABILITY MAP (IMPORTANT)
    st.markdown('<div class="section-title">🔥 Prediction Probability Map</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    im = ax.imshow(pred, cmap="inferno")
    plt.colorbar(im, ax=ax)
    ax.axis("off")
    st.pyplot(fig)

    # SAVE TEMP IMAGES FOR PDF
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "original.png")
        mask_path = os.path.join(tmpdir, "mask.png")
        overlay_path = os.path.join(tmpdir, "overlay.png")

        image_resized.save(orig_path)
        Image.fromarray(pred_binary * 255).save(mask_path)

        plt.figure()
        plt.imshow(image_resized)
        plt.imshow(pred_binary, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.savefig(overlay_path, bbox_inches="tight")
        plt.close()

        # PDF REPORT
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "Oil Spill Detection Report", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, f"Timestamp: {timestamp}", ln=True)
        pdf.cell(0, 10, f"Decision: {decision}", ln=True)
        pdf.cell(0, 10, f"Oil Spill Percentage: {oil_percentage:.2f}%", ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, "Original Image:", ln=True)
        pdf.image(orig_path, w=170)
        pdf.ln(5)

        pdf.cell(0, 10, "Predicted Mask:", ln=True)
        pdf.image(mask_path, w=170)
        pdf.ln(5)

        pdf.cell(0, 10, "Overlay Visualization:", ln=True)
        pdf.image(overlay_path, w=170)

        pdf_bytes = pdf.output(dest="S").encode("latin1")

    st.download_button(
        label="⬇️ Download Full PDF Report",
        data=pdf_bytes,
        file_name="oil_spill_report.pdf",
        mime="application/pdf"
    )


# PREDICTION HISTORY DISPLAY
st.markdown('<div class="section-title">🕒 Prediction History (Current Session)</div>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info("No predictions made yet.")
else:
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(
            f"**{i}.** {item['timestamp']} | {item['decision']} | "
            f"Oil Area: {item['oil_percentage']:.2f}%"
        )

# FOOTER
st.markdown("""
<div class="footer">

</div>
""", unsafe_allow_html=True)


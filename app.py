import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from skimage.feature import graycomatrix, graycoprops
import tempfile


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Liver MRI Cirrhosis Screening", page_icon="🧬", layout="centered")

# ---------- PREMIUM UI STYLE ----------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #4e54c8, #8f94fb); }

.stApp {
    background: rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 14px;
    backdrop-filter: blur(12px);
}

label, h1, h2, h3 {
    color: white !important;
    font-weight: 700;
}

.stButton>button {
    background: linear-gradient(90deg, #00F260, #0575E6);
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 12px;
    transition: 0.25s;
}
.stButton>button:hover { transform: scale(1.08); }

.result-box {
    background: rgba(255,255,255,0.32);
    padding: 14px;
    border-radius: 12px;
    border-left: 6px solid white;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
MODEL_PATH = "RandomForest_Cirrhosis.joblib"
model = joblib.load(MODEL_PATH)


# ---------- FEATURE FUNCTION ----------
def extract_features(slice_img):
    slice_img = (slice_img / np.max(slice_img) * 255).astype(np.uint8)
    glcm = graycomatrix(slice_img, distances=[1], angles=[0], symmetric=True, normed=True, levels=256)

    return [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'ASM')[0][0],
        graycoprops(glcm, 'energy')[0][0]
    ]


# ---------- UI HEADER ----------
st.title("🩺 AI Liver MRI Cirrhosis Screening")
st.caption("⚠ Research-use only — Not a substitute for medical diagnosis.")


# ---------------- FORM ----------------
with st.form("user_form"):

    st.subheader("👤 Patient Details")

    name = st.text_input("Enter patient name")
    age = st.text_input("Enter patient age")
    patient_id = st.text_input("Enter Patient ID")

    st.subheader("📁 Upload MRI Scans")
    t1_file = st.file_uploader("Upload T1 MRI File (.nii/.nii.gz)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Upload T2 MRI File (.nii/.nii.gz)", type=["nii", "nii.gz"])

    btn = st.form_submit_button("🔍 Run AI Analysis")


# ---------------- PROCESSING ----------------
if btn:

    if not name or not age or not patient_id:
        st.error("❌ Please fill patient details fully.")
        st.stop()

    if not t1_file or not t2_file:
        st.error("❌ Please upload BOTH T1 and T2 MRI files.")
        st.stop()

    st.info("⏳ Processing MRI scans... Please wait.")

    # Load using correct method
    try:
        t1 = nib.load(BytesIO(t1_file.getvalue())).get_fdata()
        t2 = nib.load(BytesIO(t2_file.getvalue())).get_fdata()
    except Exception as e:
        st.error(f"❌ Error reading MRI files. Not valid NIfTI format.\nDetails: {e}")
        st.stop()

    total_slices = min(t1.shape[2], t2.shape[2])
    predictions = []

    progress = st.progress(0)

    for i in range(total_slices):
        combined = (t1[:, :, i] + t2[:, :, i]) / 2

        if np.mean(combined) < 5:  # Skip black slices
            continue

        features = extract_features(combined)
        pred = model.predict([features])[0]
        predictions.append(pred)

        progress.progress(int(i / total_slices * 100))

    # Slice count
    result_count = {label: predictions.count(label) for label in ["Healthy", "Cirrhosis", "Borderline"]}

    # Final Decision Logic
    total = sum(result_count.values())
    healthy_ratio = result_count["Healthy"] / total
    cirr_ratio = result_count["Cirrhosis"] / total

    if cirr_ratio > 0.7:
        final = "Cirrhosis"
        color = "#ff4d4d"
        icon = "🔴"
    elif healthy_ratio > 0.7:
        final = "Healthy"
        color = "#4dff7a"
        icon = "🟢"
    else:
        final = "Borderline"
        color = "#ffe066"
        icon = "🟡"

    # ---------- RESULT ----------
    st.markdown(
        f'<div class="result-box" style="border-left-color:{color}"><h2>{icon} Result: {final}</h2></div>',
        unsafe_allow_html=True
    )

    # Graph
    st.subheader("📊 Slice Classification Summary")
    fig, ax = plt.subplots()
    ax.bar(result_count.keys(), result_count.values(), color=["green", "red", "orange"])
    st.pyplot(fig)

    # ---------- PDF EXPORT ----------
    st.subheader("📄 Download Report")

    pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    c = canvas.Canvas(pdf, pagesize=A4)
    c.drawString(50, 800, "AI Liver MRI Report")
    c.drawString(50, 780, f"Patient Name: {name}")
    c.drawString(50, 760, f"Age: {age}")
    c.drawString(50, 740, f"Patient ID: {patient_id}")
    c.drawString(50, 720, f"Diagnosis: {final}")
    c.drawString(50, 700, f"Prediction Breakdown: {result_count}")
    c.drawString(50, 660, "⚠ AI-based prediction. Not a medical substitute.")
    c.save()

    with open(pdf, "rb") as f:
        st.download_button("📥 Download PDF Report", f, file_name=f"{patient_id}_MRI_Report.pdf")


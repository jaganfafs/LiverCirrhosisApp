import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from skimage.feature import graycomatrix, graycoprops

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AI Liver MRI Cirrhosis Screening",
    layout="centered",
    page_icon="🧬"
)

# ---------------------- LIGHT THEME ----------------------
custom_css = """
<style>
    body {
        background: linear-gradient(135deg, #f5f7ff, #e6ecff);
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #ffffff;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.08);
    }
    .result-box {
        background: rgba(248, 249, 255, 0.95);
        padding: 16px;
        border-radius: 12px;
        border-left: 6px solid #4f46e5;
        margin-top: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #22c55e);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        transition: 0.25s;
    }
    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 8px 20px rgba(79,70,229,0.35);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("## 🩺 AI Liver MRI Cirrhosis Screening")
st.caption("⚠ Research-use only — Not a substitute for clinical diagnosis.")

# ---------------------- LOAD MODEL ----------------------
MODEL_PATH = "RandomForest_Cirrhosis.joblib"
model = joblib.load(MODEL_PATH)

# ---------------------- HELPER: LOAD NIFTI FROM UPLOAD ----------------------
def load_nifti_from_upload(uploaded_file):
    """
    Save uploaded file to a temporary path and load with nibabel.
    """
    if uploaded_file is None:
        return None

    if uploaded_file.name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    img = nib.load(tmp_path)
    data = img.get_fdata()

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return data

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_features(slice_img):
    # Handle completely flat slices safely
    max_val = np.max(slice_img)
    if max_val <= 0:
        # return dummy features (model will still output something)
        return [0.0, 0.0, 0.0, 0.0]

    slice_img = (slice_img / max_val * 255).astype(np.uint8)
    glcm = graycomatrix(
        slice_img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    return [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'ASM')[0][0],
        graycoprops(glcm, 'energy')[0][0]
    ]

# ---------------------- PATIENT FORM ----------------------
with st.form("patient_form"):
    st.subheader("👤 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Enter your name")
        patient_id = st.text_input("Enter Patient ID")

    with col2:
        age = st.text_input("Enter your age")
        scan_type = st.selectbox("Scan Type", ["MRI - Liver (T1 & T2)"])

    st.subheader("📁 Upload MRI Scans")
    t1_file = st.file_uploader("Upload T1 file (.nii / .nii.gz)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Upload T2 file (.nii / .nii.gz)", type=["nii", "nii.gz"])

    run_button = st.form_submit_button("🔍 Run AI Analysis")

# ---------------------- PROCESSING ----------------------
if run_button:
    # Basic validation
    if not patient_name.strip() or not patient_id.strip() or not age.strip():
        st.error("❌ Please enter patient name, ID, and age.")
        st.stop()

    if t1_file is None or t2_file is None:
        st.error("❌ Please upload BOTH T1 and T2 MRI files.")
        st.stop()

    st.info("⏳ Processing MRI scans... Please wait.")

    # Load MRI volumes safely
    try:
        t1_volume = load_nifti_from_upload(t1_file)
        t2_volume = load_nifti_from_upload(t2_file)
    except Exception as e:
        st.error(f"❌ Error reading MRI files. Ensure they are valid NIfTI images.\n\nDetails: {e}")
        st.stop()

    if t1_volume is None or t2_volume is None:
        st.error("❌ Could not load MRI data from uploaded files.")
        st.stop()

    # Ensure same number of slices
    total_slices = min(t1_volume.shape[2], t2_volume.shape[2])

    predictions = []
    progress = st.progress(0)

    for i in range(total_slices):
        combined_slice = (t1_volume[:, :, i] + t2_volume[:, :, i]) / 2.0

        # Only skip if slice is literally all zeros
        if np.all(combined_slice == 0):
            continue

        feats = extract_features(combined_slice)
        pred = model.predict([feats])[0]
        predictions.append(pred)

        progress.progress(int((i + 1) / total_slices * 100))

    if len(predictions) == 0:
        st.error("No valid slices found in the uploaded MRIs (all slices appear empty).")
        st.stop()

    # Count predictions
    counts = {
        "Healthy": predictions.count("Healthy"),
        "Cirrhosis": predictions.count("Cirrhosis"),
        "Borderline": predictions.count("Borderline")
    }
    total_used = sum(counts.values())

    # Ratios
    healthy_ratio = counts["Healthy"] / total_used if total_used > 0 else 0
    cirr_ratio = counts["Cirrhosis"] / total_used if total_used > 0 else 0

    # Final decision (borderline rule)
    if cirr_ratio > 0.7:
        final_result = "Cirrhosis"
        color = "#ff4d4d"
        icon = "🔴"
    elif healthy_ratio > 0.7:
        final_result = "Healthy"
        color = "#22c55e"
        icon = "🟢"
    else:
        final_result = "Borderline"
        color = "#facc15"
        icon = "🟡"

    # ---------------------- SHOW RESULT ----------------------
    st.markdown(
        f"""
        <div class="result-box" style="border-left-color:{color}">
            <h3>{icon} Result: {final_result}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Slice distribution chart
    st.subheader("📊 Slice Classification Breakdown")
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color=["green", "red", "orange"])
    ax.set_ylabel("Number of Slices")
    ax.set_xlabel("Class")
    st.pyplot(fig)

    # ---------------------- PDF REPORT ----------------------
    st.subheader("📄 Download Report")

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=A4)

    c.drawString(50, 800, "AI Liver MRI Screening Report")
    c.drawString(50, 780, f"Patient Name: {patient_name}")
    c.drawString(50, 760, f"Patient ID: {patient_id}")
    c.drawString(50, 740, f"Age: {age}")
    c.drawString(50, 720, f"Scan Type: {scan_type}")
    c.drawString(50, 690, f"Final Result: {final_result}")
    c.drawString(
        50, 670,
        f"Slices - Healthy: {counts['Healthy']}, Cirrhosis: {counts['Cirrhosis']}, Borderline: {counts['Borderline']}"
    )
    c.drawString(50, 640, "Disclaimer: This AI tool is for research/education only,")
    c.drawString(50, 625, "and is NOT a substitute for professional medical diagnosis.")
    c.save()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=f"{patient_id}_MRI_Report.pdf",
            mime="application/pdf"
        )


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


# ---------------------- UI CONFIG ----------------------
st.set_page_config(page_title="AI Liver MRI Cirrhosis Screening", page_icon="🧬", layout="centered")

# ---------------------- THEME ----------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #4e54c8, #8f94fb);
}
.stApp {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px;
}
label, h2, h3 {
    color: white !important;
    font-weight: 600;
}
.stButton>button {
    background: linear-gradient(90deg, #00F260, #0575E6);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
.result-box {
    background: rgba(255,255,255,0.3);
    border-radius: 10px;
    padding: 15px;
    border-left: 6px solid white;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- MODEL LOADING ----------------------
MODEL_PATH = "RandomForest_Cirrhosis.joblib"  # Must match GitHub file
model = joblib.load(MODEL_PATH)


# ---------------------- FEATURE EXTRACTION ----------------------
def extract_features(slice_img):
    slice_img = (slice_img / np.max(slice_img) * 255).astype(np.uint8)
    glcm = graycomatrix(slice_img, distances=[1], angles=[0], symmetric=True, normed=True, levels=256)
    return [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'ASM')[0][0],
        graycoprops(glcm, 'energy')[0][0]
    ]


# ---------------------- HEADER ----------------------
st.title("🩺 AI Liver MRI Cirrhosis Screening")
st.caption("⚠ Research-use only — Not a substitute for clinical diagnosis.")


# ---------------------- FORM ----------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Patient Name")
        patient_id = st.text_input("Patient ID")

    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        scan_type = st.selectbox("Scan Type", ["MRI - Liver (T1 & T2)"])

    st.subheader("Upload MRI Scans (Both Required)")
    t1_file = st.file_uploader("Upload T1 MRI (.nii or .nii.gz)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Upload T2 MRI (.nii or .nii.gz)", type=["nii", "nii.gz"])

    run = st.form_submit_button("🔍 Run AI Analysis")


# ---------------------- PROCESSING ----------------------
if run:

    # Validate Inputs
    if not t1_file or not t2_file or patient_name.strip() == "" or patient_id.strip() == "":
        st.error("❌ Please fill all fields and upload both scans.")
        st.stop()

    st.info("⏳ Processing MRI scans... Please wait.")

    def load_mri(uploaded):
        return nib.load(BytesIO(uploaded.read())).get_fdata()

    try:
        t1_volume = load_mri(t1_file)
        t2_volume = load_mri(t2_file)
    except:
        st.error("❌ Error reading MRI files. Ensure they are valid NIfTI format.")
        st.stop()

    predictions = []
    total_slices = min(t1_volume.shape[2], t2_volume.shape[2])

    progress = st.progress(0)

    for i in range(total_slices):
        slice_pair = (t1_volume[:, :, i] + t2_volume[:, :, i]) / 2  # Fusion

        # Skip blank slices
        if np.mean(slice_pair) < 1:
            continue

        features = extract_features(slice_pair)
        pred = model.predict([features])[0]
        predictions.append(pred)

        progress.progress(int((i / total_slices) * 100))

    # Slice Stats
    counts = {k: predictions.count(k) for k in ["Healthy", "Cirrhosis", "Borderline"]}
    total = sum(counts.values())

    # Prediction Rule
    healthy_ratio = counts["Healthy"] / total
    cirr_ratio = counts["Cirrhosis"] / total

    if cirr_ratio > 0.7:
        final_result = "Cirrhosis"
        color = "#ff4d4d"
    elif healthy_ratio > 0.7:
        final_result = "Healthy"
        color = "#4dff7a"
    else:
        final_result = "Borderline"
        color = "#ffe066"


    # ---------------------- DISPLAY RESULT ----------------------
    st.markdown(f"""
    <div class="result-box" style="border-left-color:{color}">
    <h2>{'🔴' if final_result=='Cirrhosis' else '🟢' if final_result=='Healthy' else '🟡'} Result: {final_result}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Graph
    st.subheader("📊 Slice Classification Breakdown")
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color=["green", "red", "yellow"])
    st.pyplot(fig)


    # ---------------------- PDF REPORT ----------------------
    st.subheader("📄 Download Report")

    pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf, pagesize=A4)

    c.drawString(50, 800, "Patient MRI Liver Report")
    c.drawString(50, 780, f"Name: {patient_name}")
    c.drawString(50, 760, f"ID: {patient_id}")
    c.drawString(50, 740, f"Age: {age}")
    c.drawString(50, 720, f"Final Diagnosis: {final_result}")
    c.drawString(50, 700, f"Slice Summary: {counts}")
    c.drawString(50, 660, "⚠ AI-based screening. Not a clinical diagnosis.")
    c.save()

    with open(pdf, "rb") as f:
        st.download_button("📥 Download PDF Report", f, f"{patient_id}_MRI_Report.pdf", mime="application/pdf")


import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
from skimage.feature import graycomatrix, graycoprops

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AI Liver MRI Cirrhosis Screening",
    layout="wide",
    page_icon="🧬"
)

# ---------------------- CUSTOM UI THEME ----------------------
custom_css = """
<style>
    body {
        background: linear-gradient(135deg, #4e54c8, #8f94fb);
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 30px;
        margin-top: 20px;
    }
    h1, h2, h3, h4, label {
        color: white !important;
    }
    .stProgress > div > div > div > div {
        background-color: #00eaff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00F260, #0575E6);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 16px;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0575E6, #00F260);
    }
    .result-box {
        background: rgba(255, 255, 255, 0.30);
        backdrop-filter: blur(10px);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.4);
        margin-top: 15px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
# 🏥 AI Liver MRI Cirrhosis Screening  
Upload MRI scans and generate an AI-assisted medical summary.

> ⚠️ Research-use only — Not a substitute for clinical diagnosis.
""")

# ---------------------- LOAD MODEL ----------------------
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_features(slice_img):
    slice_img = (slice_img / np.max(slice_img) * 255).astype(np.uint8)
    glcm = graycomatrix(slice_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    return [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'ASM')[0][0],
        graycoprops(glcm, 'energy')[0][0]
    ]

# ---------------------- USER INPUT FORM ----------------------
with st.form("patient_form"):
    st.subheader("📌 Patient Information")
    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    uploaded_mri = st.file_uploader("Upload MRI (.nii / .nii.gz)", type=["nii", "nii.gz"])
    
    run_button = st.form_submit_button("🔍 Run AI Analysis")

# ---------------------- PROCESSING ----------------------
if run_button and uploaded_mri:
    
    st.toast("Processing MRI volume…")
    mri = nib.load(uploaded_mri)
    volume = np.array(mri.get_fdata())

    predictions = []
    total_slices = volume.shape[2]
    progress = st.progress(0)

    for i in range(total_slices):
        slice_img = volume[:, :, i]
        if np.mean(slice_img) < 1:
            continue
        
        features = extract_features(slice_img)
        pred = model.predict([features])[0]
        predictions.append(pred)
        
        progress.progress(int((i / total_slices) * 100))

    st.success("Analysis Completed!")

    # Count slice-level predictions
    unique, counts = np.unique(predictions, return_counts=True)
    result_dict = dict(zip(unique, counts))
    
    healthy = result_dict.get("Healthy", 0)
    cirrhosis = result_dict.get("Cirrhosis", 0)
    borderline = result_dict.get("Borderline", 0)
    total = healthy + cirrhosis + borderline

    # ---------------------- BORDERLINE RULE (#3) ----------------------
    if total == 0:
        final_result = "Error"
    else:
        h_ratio = healthy / total
        c_ratio = cirrhosis / total

        if c_ratio > 0.7:
            final_result = "Cirrhosis"
        elif h_ratio > 0.7:
            final_result = "Healthy"
        else:
            final_result = "Borderline"

    # ---------------------- DISPLAY RESULT ----------------------
    if final_result == "Healthy":
        st.markdown(f"""
        <div class="result-box" style="border-left: 6px solid #4dff7a;">
        <h2>🟢 Result: Healthy Liver Pattern</h2>
        The scan does not show significant features of cirrhosis.
        </div>
        """, unsafe_allow_html=True)

    elif final_result == "Cirrhosis":
        st.markdown(f"""
        <div class="result-box" style="border-left: 6px solid #ff4d4d;">
        <h2>🔴 Result: Cirrhosis Detected</h2>
        MRI features indicate possible chronic fibrosis of the liver.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box" style="border-left: 6px solid #ffe066;">
        <h2>🟡 Result: Borderline Case</h2>
        The scan is inconclusive. Further clinical review is advised.
        </div>
        """, unsafe_allow_html=True)

    # ---------------------- GRAPH ----------------------
    st.subheader("📊 Slice Classification Breakdown")
    fig, ax = plt.subplots()
    ax.bar(result_dict.keys(), result_dict.values(), color=["green","red","yellow"])
    st.pyplot(fig)

    # ---------------------- PDF REPORT ----------------------
    st.subheader("📄 Download Report")
    
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_file, pagesize=A4)

    c.drawString(50, 800, "AI Liver MRI Screening Report")
    c.drawString(50, 770, f"Patient Name: {patient_name}")
    c.drawString(50, 750, f"Patient ID: {patient_id}")
    c.drawString(50, 730, f"Age: {age}")
    c.drawString(50, 700, f"Final Result: {final_result}")
    c.drawString(50, 670, f"Healthy Slices: {healthy}")
    c.drawString(50, 650, f"Cirrhosis Slices: {cirrhosis}")
    c.drawString(50, 630, f"Borderline Slices: {borderline}")
    c.drawString(50, 600, "Disclaimer: This AI tool is for research use only.")
    
    c.save()

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Report",
            data=f,
            file_name=f"{patient_id}_MRI_Report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload MRI and fill details to begin analysis.")

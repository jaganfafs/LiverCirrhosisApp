import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
import re
import cv2
from PIL import Image

import torch
import timm
from torchvision import transforms
from skimage.transform import resize

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="AI Liver MRI Cirrhosis Screening",
    layout="centered",
    page_icon="🧬"
)

# Path to RF model (joblib version of your original .pkl)
MODEL_PATH = "RandomForest_Cirrhosis.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mean-probability thresholds (same as in your notebook)
LOWER_THRESHOLD = 0.455   # ~45.5%
UPPER_THRESHOLD = 0.475   # ~47.5%
SLICE_INFO_THRESHOLD = 0.465   # for slice-wise count (not final decision)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_3ch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# ---------------------- THEME ----------------------
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


# ---------------------- CACHED MODELS ----------------------
@st.cache_resource
def load_vit_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.head = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

vit_model = load_vit_model()


@st.cache_resource
def load_rf_model():
    return joblib.load(MODEL_PATH)

rf_model = load_rf_model()


# ---------------------- HELPERS (from your notebook, adapted) ----------------------
def get_orig_name(uploaded_file):
    """Get original filename from Streamlit UploadedFile."""
    return os.path.basename(uploaded_file.name)


def extract_patient_id(filename):
    """
    Heuristic to extract patient ID from filename.
    e.g. 'patient01_T1.nii.gz' -> 'patient01'
    """
    base = os.path.basename(filename).lower()
    base = re.sub(r"\\.nii(\\.gz)?$", "", base)
    tokens = re.split(r"[_\\-\\.]", base)
    filtered = [t for t in tokens if t not in ["t1", "t2", "t1w", "t2w"] and t != ""]
    if not filtered:
        return base
    return "_".join(filtered)


def validate_modalities_and_patient(t1_file, t2_file):
    """
    - Ensure both files present
    - Check T1 slot isn't T2 and vice versa (by filename)
    - Check both look like same patient (simple ID heuristic)
    """
    if t1_file is None or t2_file is None:
        return False, "⚠️ Please upload both **T1** and **T2** MRI volumes."

    t1_name = get_orig_name(t1_file)
    t2_name = get_orig_name(t2_file)

    t1_lower = t1_name.lower()
    t2_lower = t2_name.lower()

    # Modality sanity check
    if "t2" in t1_lower and "t1" not in t1_lower:
        return False, f"❌ It looks like you uploaded a **T2** file (`{t1_name}`) in the **T1** slot. Please upload the correct T1 file."
    if "t1" in t2_lower and "t2" not in t2_lower:
        return False, f"❌ It looks like you uploaded a **T1** file (`{t2_name}`) in the **T2** slot. Please upload the correct T2 file."

    # Same-patient heuristic
    pid_t1 = extract_patient_id(t1_name)
    pid_t2 = extract_patient_id(t2_name)

    if pid_t1 != pid_t2:
        return False, (
            f"⚠️ The uploaded files seem to belong to **different patients**:\n"
            f"- T1 file: `{t1_name}` → ID: `{pid_t1}`\n"
            f"- T2 file: `{t2_name}` → ID: `{pid_t2}`\n\n"
            "Please ensure T1 and T2 are from the **same patient**."
        )

    return True, None


def load_nifti_from_upload(uploaded_file):
    """Save uploaded file to temp path and load with nibabel."""
    if uploaded_file is None:
        return None

    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    vol = nib.load(tmp_path).get_fdata().astype(np.float32)

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return vol


def nlm_denoise(slice_img):
    img = (slice_img * 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(
        img, None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )
    return den.astype(np.float32) / 255.0


def preprocess_slice(sl):
    if sl.max() - sl.min() < 1e-6:
        sln = np.zeros_like(sl)
    else:
        sln = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    sln = nlm_denoise(sln)
    sln = resize(sln, (224, 224), preserve_range=True).astype(np.float32)
    return sln


def vit_extract_batch(slices, progress_callback=None, start=0.4, end=0.8):
    batch = []
    total = len(slices)
    for idx, s in enumerate(slices):
        img = (s * 255).astype(np.uint8)
        s_rgb = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(s_rgb)
        t3 = transform_3ch(pil)
        batch.append(t3)

        if progress_callback is not None and total > 0:
            frac = start + (end - start) * (idx + 1) / total
            progress_callback(frac)

    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        feats = vit_model(xb)
    return feats.cpu().numpy()


def fuse_features(t1_feats, t2_feats):
    L = min(len(t1_feats), len(t2_feats))
    return np.concatenate([t1_feats[:L], t2_feats[:L]], axis=1)


# ---------------------- PATIENT FORM UI ----------------------
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


# ---------------------- MAIN PROCESSING ----------------------
if run_button:
    # Basic validation of patient info
    if not patient_name.strip() or not patient_id.strip() or not age.strip():
        st.error("❌ Please enter patient name, ID, and age.")
        st.stop()

    if t1_file is None or t2_file is None:
        st.error("❌ Please upload BOTH T1 and T2 MRI files.")
        st.stop()

    progress_bar = st.progress(0.0)
    progress_bar.progress(0.05)

    ok, msg = validate_modalities_and_patient(t1_file, t2_file)
    if not ok:
        st.error(msg)
        st.stop()

    # Load volumes
    progress_bar.progress(0.15)
    try:
        t1 = load_nifti_from_upload(t1_file)
        t2 = load_nifti_from_upload(t2_file)
    except Exception as e:
        st.error(f"❌ Error reading MRI files. Ensure they are valid NIfTI images.\n\nDetails: {e}")
        st.stop()

    progress_bar.progress(0.25)

    # Preprocess slices
    n_slices = min(t1.shape[2], t2.shape[2])
    t1_list, t2_list = [], []
    for i in range(n_slices):
        t1_list.append(preprocess_slice(t1[:, :, i]))
        t2_list.append(preprocess_slice(t2[:, :, i]))
        if i % max(1, n_slices // 10) == 0:
            frac = 0.25 + 0.15 * (i + 1) / n_slices
            progress_bar.progress(frac)

    # ViT feature extraction
    t1_feats = vit_extract_batch(t1_list, progress_callback=progress_bar.progress, start=0.4, end=0.6)
    t2_feats = vit_extract_batch(t2_list, progress_callback=progress_bar.progress, start=0.6, end=0.8)

    progress_bar.progress(0.85)

    # Fuse features and run RF
    fused = fuse_features(t1_feats, t2_feats)
    probs = rf_model.predict_proba(fused)[:, 1]  # prob of cirrhosis per slice

    final_prob = probs.mean()
    num_slices = len(probs)

    slice_cirr_mask = probs >= SLICE_INFO_THRESHOLD
    slices_cirr = int(slice_cirr_mask.sum())
    slices_healthy = num_slices - slices_cirr

    # --- Decision logic (same as your notebook) ---
    if final_prob < LOWER_THRESHOLD:
        final_label = "Healthy"
        decision_note = (
            "The AI model's estimated cirrhosis probability is **low** and falls below the "
            f"predefined threshold of {LOWER_THRESHOLD:.3f}. Within the limitations of this model, "
            "the liver MRI appears more consistent with a **non-cirrhotic** pattern.\n\n"
            "However, this is an assistive tool only. Final interpretation should be made by a qualified clinician."
        )
        color = "#22c55e"
        icon = "🟢"
    elif final_prob > UPPER_THRESHOLD:
        final_label = "Cirrhosis"
        decision_note = (
            "The AI model's estimated cirrhosis probability is **elevated** and exceeds the "
            f"predefined threshold of {UPPER_THRESHOLD:.3f}. Within the limitations of this model, "
            "the liver MRI appears more consistent with a **cirrhotic** pattern.\n\n"
            "Please correlate with clinical findings, laboratory results, and expert radiological opinion."
        )
        color = "#ff4d4d"
        icon = "🔴"
    else:
        final_label = "Borderline / Inconclusive"
        decision_note = (
            "The AI model's estimated cirrhosis probability lies within a **borderline range** "
            f"({LOWER_THRESHOLD:.3f}–{UPPER_THRESHOLD:.3f}). In this zone, the model cannot reliably "
            "differentiate between cirrhotic and non-cirrhotic liver patterns.\n\n"
            "👉 **Professional Recommendation:**\n"
            "- This case should be considered **inconclusive** from the AI perspective.\n"
            "- Please seek a detailed evaluation by a hepatologist or radiologist.\n"
            "- Additional investigations (e.g., LFTs, elastography, biopsy, follow-up imaging) "
            "may be appropriate based on clinical judgment."
        )
        color = "#facc15"
        icon = "🟡"

    progress_bar.progress(1.0)

    # ---------------------- DISPLAY RESULT ----------------------
    st.markdown(
        f"""
        <div class="result-box" style="border-left-color:{color}">
            <h3>{icon} Result: {final_label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
**Mean estimated cirrhosis probability:** `{final_prob*100:.2f}%`

- Healthy if *p* < {LOWER_THRESHOLD:.3f}  
- Borderline / Inconclusive if {LOWER_THRESHOLD:.3f} ≤ *p* ≤ {UPPER_THRESHOLD:.3f}  
- Cirrhosis if *p* > {UPPER_THRESHOLD:.3f}

{decision_note}
        """
    )

    # ---------------------- SLICE BAR CHART ----------------------
    st.subheader("📊 Slice Classification Breakdown (using slice threshold)")
    counts = {
        "Healthy": slices_healthy,
        "Cirrhosis": slices_cirr,
        "Borderline": 0
    }

    fig, ax = plt.subplots()
    ax.bar(["Healthy", "Cirrhosis", "Borderline"],
           [counts["Healthy"], counts["Cirrhosis"], counts["Borderline"]],
           color=["green", "red", "orange"])
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
    c.drawString(50, 700, f"Final Result: {final_label}")
    c.drawString(50, 680, f"Mean cirrhosis probability: {final_prob*100:.2f}%")
    c.drawString(
        50, 660,
        f"Slices leaning Healthy: {slices_healthy}, Slices leaning Cirrhosis: {slices_cirr}"
    )
    c.drawString(50, 630, "Disclaimer: This AI tool is for research/education only,")
    c.drawString(50, 615, "and is NOT a substitute for professional medical diagnosis.")
    c.save()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=f"{patient_id}_MRI_Report.pdf",
            mime="application/pdf"
        )


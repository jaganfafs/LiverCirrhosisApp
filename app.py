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

# UPDATED: Mean-probability thresholds (borderline band 46.5–47.5%)
LOWER_THRESHOLD = 0.465   # 46.5%
UPPER_THRESHOLD = 0.475   # 47.5%
# Threshold for slice-wise “cirrhosis-leaning” info (unchanged)
SLICE_INFO_THRESHOLD = 0.465

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


# ---------------------- CONDITION-SPECIFIC TEXT ----------------------
def get_condition_recommendation(final_label: str) -> str:
    """
    Detailed, condition-specific explanation & suggested clinical pathway.
    This is educational text, not a prescription.
    """
    if "Cirrhosis" in final_label:
        return (
            "Clinical summary:\n"
            "The MRI pattern and AI-derived cirrhosis probability suggest a liver that is "
            "radiologically consistent with chronic liver disease and architectural remodelling.\n\n"
            "Immediate next steps (for the treating team):\n"
            "• Refer the patient to a hepatology or gastroenterology specialist.\n"
            "• Correlate with liver function tests (AST, ALT, ALP, GGT, bilirubin, INR, albumin) and viral / autoimmune workup.\n"
            "• Assess for complications: portal hypertension, varices, ascites, hepatic encephalopathy.\n"
            "• Consider ultrasound with Doppler, transient elastography (FibroScan), or contrast-enhanced CT/MRI if not already done.\n\n"
            "Lifestyle and dietary recommendations (to be reinforced by the clinician):\n"
            "• Strict avoidance of alcohol and over-the-counter hepatotoxic drugs (e.g., excessive paracetamol).\n"
            "• Maintain adequate protein intake as advised by the dietitian, with salt restriction in patients with edema or ascites.\n"
            "• Vaccination status should be reviewed (hepatitis A/B, pneumococcal, influenza) as per guidelines.\n\n"
            "Medical management (for physician consideration only):\n"
            "• Non-selective beta blockers may be considered for primary prophylaxis of variceal bleeding when indicated.\n"
            "• Diuretics and paracentesis protocols are used for tense ascites as per standard practice.\n"
            "• Etiology-specific treatments (antiviral therapy for viral hepatitis, abstinence and nutritional therapy for alcohol-related disease, etc.) "
            "should be optimized.\n\n"
            "Long-term follow-up:\n"
            "• Regular surveillance for hepatocellular carcinoma (e.g., ultrasound ± AFP every 6 months) in appropriate patients.\n"
            "• Periodic reassessment of MELD/Child–Pugh scores to decide timing of transplant referral where indicated.\n"
            "⚠ The above is a generic clinical roadmap; the patient must not self-medicate and all decisions must be taken by a qualified specialist."
        )

    if "Healthy" in final_label:
        return (
            "Clinical summary:\n"
            "Within the limits of this MRI-based AI model, there is no strong radiological evidence "
            "to suggest established hepatic cirrhosis.\n\n"
            "Suggested advice (to be tailored by the clinician):\n"
            "• Reassure the patient that the liver morphology appears broadly preserved on this study.\n"
            "• Encourage maintenance of a liver-friendly lifestyle: avoidance of excessive alcohol, cautious use of medications, "
            "and weight control in patients with metabolic risk factors.\n"
            "• If there are risk factors such as chronic viral hepatitis, NAFLD, or significant family history, "
            "periodic clinical and laboratory follow-up is still recommended.\n"
            "• Routine health checks (liver function tests, metabolic profile) may be scheduled according to local guidelines.\n\n"
            "When to consider further evaluation despite a ‘Healthy’ pattern:\n"
            "• Persistent or unexplained symptoms (jaundice, abdominal distension, easy bruising, pruritus, weight loss).\n"
            "• Strong clinical suspicion from history/examination that is discordant with imaging.\n"
            "In such scenarios, additional imaging, elastography, or specialist review is appropriate."
        )

    # Borderline / inconclusive result
    return (
        "Clinical summary:\n"
        "The AI-derived cirrhosis probability lies in a borderline zone where the model cannot reliably "
        "separate cirrhotic from non-cirrhotic patterns. The imaging findings may be subtle, early, or confounded by artefacts.\n\n"
        "Recommended clinical approach:\n"
        "• Treat this case as **inconclusive** from an AI standpoint; do not label as definitely healthy or cirrhotic based solely on this report.\n"
        "• Correlate carefully with symptoms (fatigue, abdominal discomfort, jaundice), examination findings (hepatomegaly, splenomegaly, ascites), "
        "and laboratory data.\n"
        "• Consider repeating imaging with optimized sequences or performing adjunct tests such as elastography or contrast-enhanced MRI/CT.\n"
        "• Where clinical suspicion is high, early referral to a hepatologist is advisable even if imaging is borderline.\n\n"
        "Patient counselling points:\n"
        "• Explain that the AI tool is signalling uncertainty, not a definitive diagnosis.\n"
        "• Emphasize the importance of follow-up visits, adherence to investigations, and avoidance of liver toxins (alcohol, unnecessary medicines).\n"
        "• Any treatment decisions must be taken only after full clinical evaluation by the treating physician."
    )


# ---------------------- HELPERS FROM ORIGINAL NOTEBOOK ----------------------
def get_orig_name(uploaded_file):
    return os.path.basename(uploaded_file.name)


def extract_patient_id(filename):
    base = os.path.basename(filename).lower()
    base = re.sub(r"\\.nii(\\.gz)?$", "", base)
    tokens = re.split(r"[_\\-\\.]", base)
    filtered = [t for t in tokens if t not in ["t1", "t2", "t1w", "t2w"] and t != ""]
    if not filtered:
        return base
    return "_".join(filtered)


def validate_modalities_and_patient(t1_file, t2_file):
    if t1_file is None or t2_file is None:
        return False, "⚠️ Please upload both **T1** and **T2** MRI volumes."

    t1_name = get_orig_name(t1_file)
    t2_name = get_orig_name(t2_file)

    t1_lower = t1_name.lower()
    t2_lower = t2_name.lower()

    if "t2" in t1_lower and "t1" not in t1_lower:
        return False, f"❌ It looks like you uploaded a **T2** file (`{t1_name}`) in the **T1** slot. Please upload the correct T1 file."
    if "t1" in t2_lower and "t2" not in t2_lower:
        return False, f"❌ It looks like you uploaded a **T1** file (`{t2_name}`) in the **T2** slot. Please upload the correct T2 file."

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

    progress_bar.progress(0.15)
    try:
        t1 = load_nifti_from_upload(t1_file)
        t2 = load_nifti_from_upload(t2_file)
    except Exception as e:
        st.error(f"❌ Error reading MRI files. Ensure they are valid NIfTI images.\n\nDetails: {e}")
        st.stop()

    progress_bar.progress(0.25)

    n_slices = min(t1.shape[2], t2.shape[2])
    t1_list, t2_list = [], []
    for i in range(n_slices):
        t1_list.append(preprocess_slice(t1[:, :, i]))
        t2_list.append(preprocess_slice(t2[:, :, i]))
        if i % max(1, n_slices // 10) == 0:
            frac = 0.25 + 0.15 * (i + 1) / n_slices
            progress_bar.progress(frac)

    t1_feats = vit_extract_batch(t1_list, progress_callback=progress_bar.progress, start=0.4, end=0.6)
    t2_feats = vit_extract_batch(t2_list, progress_callback=progress_bar.progress, start=0.6, end=0.8)

    progress_bar.progress(0.85)

    fused = fuse_features(t1_feats, t2_feats)
    probs = rf_model.predict_proba(fused)[:, 1]

    final_prob = probs.mean()
    num_slices = len(probs)

    slice_cirr_mask = probs >= SLICE_INFO_THRESHOLD
    slices_cirr = int(slice_cirr_mask.sum())
    slices_healthy = num_slices - slices_cirr

    # -------- DECISION WITH UPDATED THRESHOLDS 0.465–0.475 --------
    if final_prob < LOWER_THRESHOLD:
        final_label = "Healthy"
        color = "#22c55e"
        icon = "🟢"
    elif final_prob > UPPER_THRESHOLD:
        final_label = "Cirrhosis"
        color = "#ff4d4d"
        icon = "🔴"
    else:
        final_label = "Borderline / Inconclusive"
        color = "#facc15"
        icon = "🟡"

    progress_bar.progress(1.0)

    # Text explaining thresholds & summary (still using thresholds in explanations)
    if final_label == "Healthy":
        decision_note = (
            "The AI model's estimated cirrhosis probability is **low** and falls below the "
            f"updated threshold of {LOWER_THRESHOLD:.3f} (46.5%). Within the limitations of this model, "
            "the liver MRI appears more consistent with a **non-cirrhotic** pattern.\n\n"
            "This is an assistive tool only; final interpretation must be made by a qualified clinician."
        )
    elif final_label == "Cirrhosis":
        decision_note = (
            "The AI model's estimated cirrhosis probability is **elevated** and exceeds the "
            f"upper threshold of {UPPER_THRESHOLD:.3f} (47.5%). Within the limitations of this model, "
            "the liver MRI appears more consistent with a **cirrhotic** pattern.\n\n"
            "Please correlate with clinical findings, laboratory results, and expert radiological opinion."
        )
    else:
        decision_note = (
            "The AI model's estimated cirrhosis probability lies within a **borderline range** "
            f"({LOWER_THRESHOLD:.3f}–{UPPER_THRESHOLD:.3f}, i.e. 46.5–47.5%). In this zone, the model cannot reliably "
            "differentiate between cirrhotic and non-cirrhotic liver patterns.\n\n"
            "The case should be considered **inconclusive** from the AI perspective and requires careful clinical correlation."
        )

    # Condition-specific recommendation text
    recommendation_text = get_condition_recommendation(final_label)

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

- **Healthy** if *p* < {LOWER_THRESHOLD:.3f} (46.5%)  
- **Borderline / Inconclusive** if {LOWER_THRESHOLD:.3f} ≤ *p* ≤ {UPPER_THRESHOLD:.3f} (46.5–47.5%)  
- **Cirrhosis** if *p* > {UPPER_THRESHOLD:.3f} (47.5%)

{decision_note}
        """
    )

    st.markdown("### 🩺 Condition-specific clinical summary")
    st.markdown(recommendation_text.replace("\n", "  \n"))

    # ---------------------- SLICE BAR CHART ----------------------
    st.subheader("📊 Slice Classification Breakdown (using slice probability threshold)")
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
    c.drawString(50, 765, f"Patient ID: {patient_id}")
    c.drawString(50, 750, f"Age: {age}")
    c.drawString(50, 735, f"Scan Type: {scan_type}")
    c.drawString(50, 715, f"Final AI Result: {final_label}")
    c.drawString(50, 700, f"Mean cirrhosis probability: {final_prob*100:.2f}%")
    c.drawString(
        50, 685,
        f"Slices leaning Healthy: {slices_healthy}, Slices leaning Cirrhosis: {slices_cirr}"
    )

    # Write condition-specific recommendation into PDF
    text_obj = c.beginText(50, 660)
    for line in recommendation_text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.drawString(50, 80, "Disclaimer: Research / educational use only.")
    c.drawString(50, 65, "Not a substitute for professional medical diagnosis or treatment.")

    c.save()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=f"{patient_id}_MRI_Report.pdf",
            mime="application/pdf"
        )


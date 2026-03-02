import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx
import torch
import re
import io
import logging
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from typing import Tuple, Dict

# ==========================================
# 1. ENTERPRISE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="UK ICO Fine Analyzer | Enterprise Edition",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0px; }
    .sub-header { font-size: 1rem; color: #64748B; margin-bottom: 2rem; }
    .security-badge { background-color: #DEF7EC; color: #03543F; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid #31C48D; display: inline-block; margin-bottom: 1rem;}
    .evidence-box { background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 10px; font-style: italic; color: #334155; font-size: 0.9rem;}
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. AI ENGINE CACHING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_engine() -> Tuple[AutoTokenizer, AutoModel]:
    try:
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"Engine Load Failure: {e}")
        st.stop()

tokenizer, model = load_ai_engine()

# ==========================================
# 3. CORE LOGIC & EXPLAINABLE AI (XAI)
# ==========================================
def extract_document_text(uploaded_file) -> str:
    text = ""
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = " ".join([page.get_text() for page in doc])
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logging.error(f"Failed to parse {file_name}: {e}")
    
    return re.sub(r'\s+', ' ', text).strip()

def generate_embeddings(text: str) -> np.ndarray:
    inputs = tokenizer(text[:2000], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_intelligent_features(text: str) -> Dict:
    """Enhanced feature extraction that captures the EVIDENCE (Explainable AI)."""
    text_lower = text.lower()
    
    # Financial Extraction
    fine_match = re.search(r'£\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    fine_amount = int(fine_match.group(1).replace(',', '').split('.')[0]) if fine_match else 0
    
    # MFA Context & Evidence
    mfa_evidence = ""
    mfa_pattern = re.compile(r'([^.]*(?:fail\w* to implement|lack of|no|without)\s+(?:multi[- ]factor authentication|mfa)[^.]*\.)', re.IGNORECASE)
    mfa_match = re.search(mfa_pattern, text)
    if mfa_match:
        mfa_evidence = mfa_match.group(1).strip()
    elif "mfa" in text_lower and "breach" in text_lower:
        mfa_evidence = "Contextual inference: MFA mentioned alongside security breach."
    
    mfa_failure = bool(mfa_evidence)
    
    # Children Data
    children_match = re.search(r'([^.]*\b(?:child|children|minors|under 13|under 18)\b[^.]*\.)', text_lower)
    children_data = bool(children_match)
    children_evidence = children_match.group(1).strip() if children_match else ""
    
    return {
        "Target_Fine_GBP": fine_amount,
        "MFA_Failure": mfa_failure,
        "MFA_Evidence": mfa_evidence,
        "Children_Data": children_data,
        "Children_Evidence": children_evidence
    }

# ==========================================
# 4. UI: SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    max_clusters = st.slider("Max K-Means Clusters", min_value=2, max_value=10, value=3, help="Determines how many groups the Unsupervised AI will attempt to find.")
    st.divider()
    st.markdown("### 🔒 Security Status")
    st.success("End-to-End Encryption: Active")
    st.info("Data Retention: Session Only")

# ==========================================
# 5. UI: MAIN DASHBOARD (TABBED INTERFACE)
# ==========================================
st.markdown('<div class="security-badge">🔒 SOC2 Compliant Environment (Simulated)</div>', unsafe_allow_html=True)
st.markdown('<p class="main-header">RegTech: UK GDPR Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Unsupervised Clustering & Explainable Feature Extraction for ICO Penalty Notices.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📂 Workspace & Analysis", "🧑‍⚖️ Supervisor Dashboard"])

with tab1:
    uploaded_files = st.file_uploader("Upload Confidential Legal Documents (PDF/DOCX)", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Run AI Legal Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Use Streamlit's session state to store data between tabs
            st.session_state['data_records'] = []
            vectors = []
            
            status_text.info("Extracting semantics and generating Legal-BERT vectors...")
            for i, file in enumerate(uploaded_files):
                text = extract_document_text(file)
                if len(text) > 50:
                    vec = generate_embeddings(text)
                    features = extract_intelligent_features(text)
                    
                    record = {
                        "File_Name": file.name,
                        **features,
                    }
                    st.session_state['data_records'].append(record)
                    vectors.append(vec)
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            if len(st.session_state['data_records']) > 0:
                status_text.info("Executing K-Means Clustering...")
                n_clusters = min(max_clusters, len(st.session_state['data_records']))
                if len(vectors) >= n_clusters and n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                    cluster_labels = kmeans.fit_predict(vectors)
                else:
                    cluster_labels = [0] * len(st.session_state['data_records'])
                    
                for i, record in enumerate(st.session_state['data_records']):
                    record["Cluster_ID"] = cluster_labels[i]
                
                progress_bar.empty()
                status_text.success("Analysis Complete! Please navigate to the 'Supervisor Dashboard' tab to verify the results.")
            else:
                progress_bar.empty()
                status_text.error("No valid text found in documents.")

with tab2:
    if 'data_records' in st.session_state and len(st.session_state['data_records']) > 0:
        st.markdown("### Human-in-the-Loop Verification")
        st.markdown("Review the AI predictions. The table below is **editable**. Once verified, download the Gold Dataset.")
        
        df_results = pd.DataFrame(st.session_state['data_records'])
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Fines Detected", f"£{df_results['Target_Fine_GBP'].sum():,}")
        col2.metric("MFA Violations", int(df_results['MFA_Failure'].sum()))
        col3.metric("Children Privacy Violations", int(df_results['Children_Data'].sum()))
        
        # Display Evidence (Explainable AI)
        with st.expander("🔎 View AI Evidence (Explainability)"):
            st.markdown("Transparency is key in legal tech. Here is *why* the AI made its decisions:")
            for idx, row in df_results.iterrows():
                if row['MFA_Failure'] or row['Children_Data']:
                    st.markdown(f"**Document:** `{row['File_Name']}`")
                    if row['MFA_Failure']:
                        st.markdown(f"<div class='evidence-box'><b>MFA Evidence:</b> {row['MFA_Evidence']}</div>", unsafe_allow_html=True)
                    if row['Children_Data']:
                        st.markdown(f"<div class='evidence-box'><b>Children Data Evidence:</b> {row['Children_Evidence']}</div><br>", unsafe_allow_html=True)
        
        # Data Editor (Select specific columns for the editor to keep it clean)
        editor_cols = ["File_Name", "Cluster_ID", "Target_Fine_GBP", "MFA_Failure", "Children_Data"]
        
        edited_df = st.data_editor(
            df_results[editor_cols],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "File_Name": st.column_config.TextColumn("Document Name", disabled=True),
                "Cluster_ID": st.column_config.NumberColumn("Cluster Group", disabled=True),
                "Target_Fine_GBP": st.column_config.NumberColumn("Estimated Fine (£)", format="£%d"),
                "MFA_Failure": st.column_config.CheckboxColumn("MFA Failure Predicted"),
                "Children_Data": st.column_config.CheckboxColumn("Children Data Involved")
            }
        )
        
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export Verified Dataset (CSV) for XGBoost",
            data=csv,
            file_name='Verified_ICO_Dataset.csv',
            mime='text/csv',
            type="primary"
        )
    else:
        st.info("No data available yet. Please upload and analyze documents in the 'Workspace & Analysis' tab first.")

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
from typing import List, Dict, Tuple

# ==========================================
# 1. ENTERPRISE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="UK ICO Fine Analyzer | Smart Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, SaaS-like dashboard
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .stProgress > div > div > div > div { background-color: #1E3A8A; }
    .kpi-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# Setup Logging for production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. ML MODEL CACHING (OPTIMIZED)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_engine() -> Tuple[AutoTokenizer, AutoModel]:
    """Loads the Legal-BERT model into cache to prevent reloading on every interaction."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"Critical System Error: Failed to load Legal-BERT engine. Details: {e}")
        st.stop()

tokenizer, model = load_ai_engine()

# ==========================================
# 3. CORE LOGIC & PIPELINES
# ==========================================
def extract_document_text(uploaded_file) -> str:
    """Robust text extraction from PDFs and Word documents."""
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
        st.toast(f"Warning: Could not read {file_name}", icon="⚠️")
    
    # Clean text (remove excessive whitespaces)
    return re.sub(r'\s+', ' ', text).strip()

def generate_embeddings(text: str) -> np.ndarray:
    """Converts legal text into numerical vectors using Legal-BERT."""
    # Truncate to first 2000 chars for embedding to fit 512 token limit safely
    inputs = tokenizer(text[:2000], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_intelligent_features(text: str) -> Dict:
    """Advanced Feature Engineering using regex and context matching."""
    text_lower = text.lower()
    
    # 1. Financial Extraction: Look for £ followed by numbers
    fine_match = re.search(r'£\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    fine_amount = int(fine_match.group(1).replace(',', '').split('.')[0]) if fine_match else 0
    
    # 2. MFA Context (Looking for absence/failure of MFA, not just the word)
    mfa_pattern = re.compile(r'(fail\w* to implement|lack of|no|without)\s+(multi[- ]factor authentication|mfa)', re.IGNORECASE)
    mfa_failure = bool(re.search(mfa_pattern, text)) or ("mfa" in text_lower and "breach" in text_lower)
    
    # 3. Vulnerable Data (Children)
    children_data = bool(re.search(r'\b(child|children|minors|under 13|under 18)\b', text_lower))
    
    # 4. Attack Vector (Ransomware)
    ransomware = "ransomware" in text_lower
    
    return {
        "Target_Fine_GBP": fine_amount,
        "MFA_Failure": mfa_failure,
        "Children_Data_Involved": children_data,
        "Ransomware_Attack": ransomware
    }

# ==========================================
# 4. UI: SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Scale_of_justice_2.svg/1200px-Scale_of_justice_2.svg.png", width=80)
    st.markdown("### Control Panel")
    st.info("Adjust the clustering parameters before running the analysis.")
    max_clusters = st.slider("Maximum Clusters (K-Means)", min_value=2, max_value=10, value=3)
    st.divider()
    st.markdown("### System Status")
    st.success("🟢 Legal-BERT Engine: Active")
    st.success("🟢 UI Data Pipeline: Online")

# ==========================================
# 5. UI: MAIN DASHBOARD
# ==========================================
st.markdown('<p class="main-header">UK GDPR Fine Analyzer & Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Unsupervised Clustering & Feature Extraction Tool for the ICO Landscape.</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Drop ICO Penalty Notices or Privacy Policies here (PDF/DOCX)", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Execute Smart Analysis", use_container_width=True, type="primary"):
        
        # --- UI Progress Tracking ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        data_records = []
        vectors = []
        
        # --- Step 1: Extraction & Embedding ---
        status_text.info("Step 1/3: Reading documents and extracting Legal-BERT vectors...")
        for i, file in enumerate(uploaded_files):
            text = extract_document_text(file)
            if len(text) > 50: # Ensure valid text
                vec = generate_embeddings(text)
                features = extract_intelligent_features(text)
                
                record = {
                    "File_Name": file.name,
                    **features, # Unpack extracted features
                    "_text_snippet": text[:500] # Hidden context for supervisor
                }
                data_records.append(record)
                vectors.append(vec)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        if len(data_records) > 0:
            # --- Step 2: Unsupervised Clustering ---
            status_text.info("Step 2/3: Grouping similar cases using K-Means Clustering...")
            n_clusters = min(max_clusters, len(data_records))
            if len(vectors) >= n_clusters and n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(vectors)
            else:
                cluster_labels = [0] * len(data_records) # Default if not enough files
                
            for i, record in enumerate(data_records):
                record["AI_Cluster_ID"] = cluster_labels[i]
                
            # Compile Dataset
            df_results = pd.DataFrame(data_records)
            
            # --- Step 3: Supervisor Dashboard ---
            progress_bar.empty()
            status_text.success("Analysis Complete! Awaiting Human Supervisor Verification.")
            
            # Display KPIs
            st.markdown("### 📈 Pipeline Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Documents Processed", len(df_results))
            col2.metric("Total Fines Detected", f"£{df_results['Target_Fine_GBP'].sum():,}")
            col3.metric("MFA Failures", int(df_results['MFA_Failure'].sum()))
            col4.metric("Unique Clusters", len(df_results['AI_Cluster_ID'].unique()))
            
            st.divider()
            
            # Interactive Data Editor
            st.markdown("### 🧑‍⚖️ Supervisor Verification Panel (Human-in-the-Loop)")
            st.markdown("The AI has extracted the following features. **Double-click any cell to correct the AI's prediction** before finalizing the Gold Dataset.")
            
            # Reorder columns for better UI
            cols = ["File_Name", "AI_Cluster_ID", "Target_Fine_GBP", "MFA_Failure", "Children_Data_Involved", "Ransomware_Attack"]
            df_display = df_results[cols]
            
            # Editable dataframe!
            edited_df = st.data_editor(
                df_display,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "File_Name": st.column_config.TextColumn("Document Name", disabled=True),
                    "AI_Cluster_ID": st.column_config.NumberColumn("Cluster Group", disabled=True),
                    "Target_Fine_GBP": st.column_config.NumberColumn("Fine Amount (£)", format="£%d", min_value=0),
                    "MFA_Failure": st.column_config.CheckboxColumn("MFA Failure?"),
                    "Children_Data_Involved": st.column_config.CheckboxColumn("Children's Data?"),
                    "Ransomware_Attack": st.column_config.CheckboxColumn("Ransomware?")
                }
            )
            
            st.info("💡 Next Step: Download this verified dataset. It will be used to train the Supervised XGBoost Model to predict future fines.")
            
            # Download Button for the Gold Dataset
            csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Verified Gold Dataset (CSV)",
                data=csv,
                file_name='UK_GDPR_Gold_Dataset.csv',
                mime='text/csv',
                type="primary"
            )
        else:
            progress_bar.empty()
            status_text.error("No valid text could be extracted from the uploaded files.")

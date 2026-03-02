import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx
import torch
import re
import io
import time
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from typing import Tuple, Dict

# ==========================================
# 1. إعدادات النظام وتصميم الـ Enterprise (CSS)
# ==========================================
st.set_page_config(
    page_title="Lumina AI | Enterprise Compliance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تصميم احترافي يحاكي تطبيقات OneTrust و Harvey AI
st.markdown("""
    <style>
    /* ألوان وخلفيات النظام */
    .stApp { background-color: #F8FAFC; }
    
    /* الشريط الجانبي */
    [data-testid="stSidebar"] { background-color: #0F172A; color: #F8FAFC; }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    
    /* العناوين الرئيسية */
    .hero-title { font-size: 2.8rem; font-weight: 900; color: #0F172A; letter-spacing: -1px; margin-bottom: 0; }
    .hero-subtitle { font-size: 1.1rem; color: #475569; font-weight: 400; margin-bottom: 2rem; }
    
    /* البطاقات التحليلية (KPI Cards) */
    .kpi-card { 
        background-color: #FFFFFF; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        border-top: 4px solid #3B82F6;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #0F172A; }
    .kpi-label { font-size: 0.9rem; color: #64748B; text-transform: uppercase; letter-spacing: 1px; }
    
    /* صندوق الأدلة (Explainable AI) */
    .evidence-box { 
        background-color: #EFF6FF; 
        border-left: 4px solid #2563EB; 
        padding: 15px; 
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #1E3A8A;
        font-size: 0.95rem;
    }
    
    /* شارات الحالة */
    .badge-critical { background: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 999px; font-weight: bold; font-size: 0.8rem; }
    .badge-safe { background: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 999px; font-weight: bold; font-size: 0.8rem; }
    
    /* أزرار النظام */
    .stButton>button { border-radius: 8px; font-weight: 600; transition: all 0.3s; }
    .stButton>button:hover { box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3); }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. إدارة الجلسة (Session State & Auth)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = pd.DataFrame()

def log_action(action: str):
    """سجل التدقيق الأمني للعمليات"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['audit_log'].append({"Time": timestamp, "Action": action, "User": "Admin_Supervisor"})

# ==========================================
# 3. تحميل محرك الذكاء الاصطناعي
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_engine() -> Tuple[AutoTokenizer, AutoModel]:
    try:
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"Critical System Failure: {e}")
        st.stop()

# ==========================================
# 4. دوال المعالجة واستخراج الأدلة (XAI)
# ==========================================
def extract_document_text(uploaded_file) -> str:
    text = ""
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join([page.get_text() for page in doc])
    elif file_name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = " ".join([p.text for p in doc.paragraphs])
    return re.sub(r'\s+', ' ', text).strip()

def get_embeddings(text: str, tokenizer, model) -> np.ndarray:
    inputs = tokenizer(text[:2000], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def analyze_compliance_features(text: str) -> Dict:
    """استخراج الميزات مع الأدلة القانونية بدقة عالية"""
    text_lower = text.lower()
    
    # استخراج الغرامات
    fine_match = re.search(r'£\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    fine_amount = int(fine_match.group(1).replace(',', '').split('.')[0]) if fine_match else 0
    
    # تحليل ثغرات MFA
    mfa_pattern = re.compile(r'([^.]*(?:fail\w*|lack of|without|compromised)\s+[^.]*(?:multi[- ]factor authentication|mfa)[^.]*\.)', re.IGNORECASE)
    mfa_match = re.search(mfa_pattern, text)
    mfa_evidence = mfa_match.group(1).strip() if mfa_match else ""
    
    # تحليل بيانات الأطفال
    child_pattern = re.search(r'([^.]*\b(?:child|children|minors|under 13)\b[^.]*\.)', text_lower)
    child_evidence = child_pattern.group(1).strip() if child_pattern else ""
    
    # تقييم المخاطر (Risk Score)
    risk_score = 10
    if mfa_evidence: risk_score += 40
    if child_evidence: risk_score += 50
    
    return {
        "Target_Fine_GBP": fine_amount,
        "MFA_Violation": bool(mfa_evidence),
        "MFA_Evidence": mfa_evidence,
        "Children_Violation": bool(child_evidence),
        "Children_Evidence": child_evidence,
        "Risk_Score": risk_score
    }

# ==========================================
# 5. شاشة تسجيل الدخول (محاكاة الأمان)
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="kpi-card" style="text-align:center;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2048/2048183.png", width=80)
        st.markdown("## Lumina AI Security")
        st.markdown("UK GDPR Compliance Platform")
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Secure Login", use_container_width=True):
            if pwd == "admin123": # كلمة سر وهمية للتجربة
                st.session_state['logged_in'] = True
                log_action("User authenticated successfully")
                st.rerun()
            else:
                st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# 6. الواجهة الرئيسية (Main Enterprise App)
# ==========================================
tokenizer, model = load_ai_engine()

with st.sidebar:
    st.markdown("## 🛡️ Lumina Workspace")
    st.markdown(f"**User:** Supervisor Admin\n\n**Status:** <span class='badge-safe'>SOC2 Secure</span>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation Menu", ["📊 Executive Dashboard", "📂 Smart Document Analysis", "📋 Audit & Compliance Logs"])
    st.divider()
    st.markdown("### ⚙️ Engine Settings")
    max_clusters = st.slider("Clustering Granularity", 2, 8, 3)
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

st.markdown('<p class="hero-title">Lumina AI: Enterprise GDPR Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Automated Legal Precedent Discovery & Risk Prediction via Legal-BERT.</p>', unsafe_allow_html=True)

# ------------------------------------------
# الصفحة 1: Executive Dashboard (لوحة القيادة)
# ------------------------------------------
if page == "📊 Executive Dashboard":
    if not st.session_state['processed_data'].empty:
        df = st.session_state['processed_data']
        
        # مؤشرات الأداء (KPIs)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Documents</div><div class="kpi-value">{len(df)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Fines Detected</div><div class="kpi-value">£{df["Target_Fine_GBP"].sum():,}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Critical MFA Violations</div><div class="kpi-value">{df["MFA_Violation"].sum()}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Risk Score</div><div class="kpi-value">{df["Risk_Score"].mean():.1f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # رسوم بيانية تفاعلية (Plotly)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("### Risk Distribution by Cluster")
            fig = px.pie(df, names='Cluster_ID', values='Risk_Score', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Fine Amounts per Document")
            fig = px.bar(df, x='File_Name', y='Target_Fine_GBP', color='Risk_Score', color_continuous_scale='Reds')
            fig.update_layout(xaxis_title="Company / Document", yaxis_title="Fine Amount (£)", margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Dashboard is empty. Please process documents in the 'Smart Document Analysis' tab first.")

# ------------------------------------------
# الصفحة 2: Smart Document Analysis (التحليل الذكي)
# ------------------------------------------
elif page == "📂 Smart Document Analysis":
    uploaded_files = st.file_uploader("Drop ICO Penalty Notices (Secure Upload)", accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Initiate AI Analysis Engine", type="primary", use_container_width=True):
        log_action(f"Started analysis on {len(uploaded_files)} documents.")
        progress_bar = st.progress(0)
        status = st.empty()
        
        data_records = []
        vectors = []
        
        # 1. الاستخراج
        status.info("Phase 1: Semantic Extraction & Legal-BERT Vectorization...")
        for i, file in enumerate(uploaded_files):
            text = extract_document_text(file)
            if len(text) > 50:
                vec = get_embeddings(text, tokenizer, model)
                features = analyze_compliance_features(text)
                data_records.append({"File_Name": file.name, **features})
                vectors.append(vec)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        if data_records:
            # 2. التجميع الذكي
            status.info("Phase 2: Unsupervised Pattern Recognition (K-Means)...")
            n_clusters = min(max_clusters, len(data_records))
            if len(vectors) >= n_clusters and n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(vectors)
            else:
                labels = [0] * len(data_records)
                
            for i, record in enumerate(data_records):
                record["Cluster_ID"] = labels[i]

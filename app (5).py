# -*- coding: utf-8 -*-
"""
===========================================================================
🛡️ AI AUTO DATA CLEANING & CORRUPTION DETECTION SYSTEM
===========================================================================
نظام متكامل لتنظيف وتحليل أي بيانات تلقائياً، كشف الفساد والأنماط الشاذة
باستخدام الذكاء الاصطناعي - يدعم جميع أنواع الملفات والبيانات

الإصدار: 3.0 (AutoML Edition)
المطور: النظام الذكي للرقابة الإدارية
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
import os
import re
import io
import base64
from datetime import datetime
import time
from collections import Counter
import hashlib
import json

warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# XGBoost للتنبؤ المتقدم
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# transformers للتحليل المتقدم للنصوص
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# ==================== مكتبات معالجة النصوص العربية ====================
try:
    from wordcloud import WordCloud, STOPWORDS
    import arabic_reshaper
    from bidi.algorithm import get_display
    import PyPDF2
    from textblob import TextBlob
    TEXT_ANALYSIS_AVAILABLE = True
except:
    TEXT_ANALYSIS_AVAILABLE = False

# ==================== إعدادات الصفحة المتقدمة ====================
st.set_page_config(
    page_title="AI Auto Data Cleaner & Auditor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.ai-audit-system.com',
        'Report a bug': "https://github.com/ai-audit/issues",
        'About': "# AI Auto Data Cleaner\nالإصدار 3.0 - يدعم أي بيانات"
    }
)

# ==================== CSS احترافي متطور (نفس السابق) ====================
PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* الأساسيات */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* خلفية داكنة متدرجة */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', 'Cairo', sans-serif;
    }
    
    /* هيدر رئيسي بتأثير زجاجي */
    .main-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0 0 40px 40px;
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
        animation: rotate 30s linear infinite;
        z-index: 0;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88 0%, #00cc88 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 30px rgba(0, 255, 136, 0.3); }
        50% { text-shadow: 0 0 50px rgba(0, 255, 136, 0.6); }
    }
    
    .main-header p {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    /* كروت زجاجية متطورة */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.02), transparent);
        transition: left 0.8s;
    }
    
    .glass-card:hover::after {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(0, 255, 136, 0.3);
        box-shadow: 0 20px 40px rgba(0, 255, 136, 0.1);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(0, 255, 136, 0.2);
        padding-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* مقاييس نيون متألقة */
    .metric-neon {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 200, 100, 0.05) 100%);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .metric-neon::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ff88, transparent, #00ff88);
        border-radius: 22px;
        z-index: -1;
        animation: borderGlow 3s linear infinite;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-neon:hover::before {
        opacity: 1;
    }
    
    @keyframes borderGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .metric-neon:hover {
        transform: scale(1.05);
        border-color: #00ff88;
    }
    
    .metric-neon-value {
        font-size: 3rem;
        font-weight: 900;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        line-height: 1.2;
    }
    
    .metric-neon-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    /* شارات متخصصة */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin: 0.3rem;
        transition: all 0.3s;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        filter: brightness(1.2);
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #00ff88, #00cc88);
        color: #0f172a;
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ff4b4b, #dc2626);
        color: white;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fbbf24, #d97706);
        color: #0f172a;
        box-shadow: 0 5px 15px rgba(251, 191, 36, 0.3);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* أزرار نيون متطورة */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc88 100%);
        color: #0f172a;
        border: none;
        border-radius: 14px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s;
        box-shadow: 0 8px 20px rgba(0, 255, 136, 0.3);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transform: rotate(45deg);
        animation: buttonShine 3s infinite;
    }
    
    @keyframes buttonShine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(0, 255, 136, 0.5);
    }
    
    /* تبويبات متطورة */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 60px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.6);
        border: none;
        transition: all 0.3s;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #00cc88) !important;
        color: #0f172a !important;
        box-shadow: 0 8px 20px rgba(0, 255, 136, 0.4);
        font-weight: 700;
    }
    
    /* شريط جانبي متطور */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-content {
        padding: 2rem 1rem;
    }
    
    /* شريط التقدم */
    .progress-container {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00cc88);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* تذييل احترافي */
    .footer {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-top: 1px solid rgba(0, 255, 136, 0.2);
        padding: 3rem 2rem;
        margin-top: 4rem;
        text-align: center;
        border-radius: 40px 40px 0 0;
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, transparent);
    }
    
    .footer h3 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #00ff88, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
    }
    
    /* أنيميشن للعناصر */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 4s ease-in-out infinite;
    }
    
    /* تنسيق الجداول */
    .dataframe {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .dataframe th {
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        font-weight: 600;
        padding: 1rem;
    }
    
    .dataframe td {
        color: rgba(255, 255, 255, 0.8);
        padding: 0.8rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* تنسيق التنبيهات */
    .alert {
        padding: 1.2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    
    .alert-success {
        background: rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.3);
        color: #00ff88;
    }
    
    .alert-warning {
        background: rgba(251, 191, 36, 0.1);
        border-color: rgba(251, 191, 36, 0.3);
        color: #fbbf24;
    }
    
    .alert-danger {
        background: rgba(255, 75, 75, 0.1);
        border-color: rgba(255, 75, 75, 0.3);
        color: #ff4b4b;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: #3b82f6;
    }
    
    /* تحسين ظهور النصوص العربية */
    .arabic-text {
        direction: rtl;
        font-family: 'Cairo', sans-serif;
    }
</style>
"""

# ==================== تطبيق CSS ====================
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة ====================
def init_session_state():
    """تهيئة جميع متغيرات الجلسة"""
    defaults = {
        'data_loaded': False,
        'original_df': None,
        'cleaned_df': None,
        'model_trained': False,
        'anomalies': None,
        'model_pack': None,
        'cleaning_report': None,
        'predictions': None,
        'text_data': [],
        'analysis_history': [],
        'theme': 'dark',
        'processing_time': 0,
        'file_info': {},
        'corruption_cases': [],
        'nlp_model': None,
        'auto_target': None,
        'data_profile': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== تحميل نموذج NLP ====================
@st.cache_resource
def load_nlp_model():
    """تحميل نموذج تحليل النصوص"""
    if TRANSFORMERS_AVAILABLE:
        try:
            return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            st.warning(f"لم يتم تحميل نموذج NLP: {str(e)}")
            return None
    return None

# ==================== دوال معالجة البيانات الشاملة ====================

def load_any_file(uploaded_file):
    """تحميل أي نوع من الملفات"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            # محاولة قراءة CSV بتشفيرات مختلفة
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except:
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension == 'txt':
            # محاولة قراءة النص كـ CSV أو كـ نص عادي
            try:
                df = pd.read_csv(uploaded_file, sep='\t|,|;', engine='python')
            except:
                # إذا فشل، نقرأ كنص عادي
                content = uploaded_file.getvalue().decode('utf-8')
                lines = content.split('\n')
                df = pd.DataFrame({'text': lines})
        
        else:
            return None, f"صيغة الملف غير مدعومة: {file_extension}"
        
        return df, None
    
    except Exception as e:
        return None, f"خطأ في قراءة الملف: {str(e)}"

def auto_detect_column_types(df):
    """اكتشاف أنواع الأعمدة تلقائياً"""
    profile = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': [],
        'id_columns': []
    }
    
    for col in df.columns:
        # محاولة تحويل النوع
        try:
            # التحقق من الأعمدة الرقمية
            if pd.api.types.is_numeric_dtype(df[col]):
                profile['numeric'].append(col)
            
            # التحقق من الأعمدة النصية
            elif pd.api.types.is_string_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                
                if unique_ratio < 0.05:  # أقل من 5% قيم فريدة
                    profile['categorical'].append(col)
                elif unique_ratio > 0.9:  # أكثر من 90% قيم فريدة
                    profile['id_columns'].append(col)
                else:
                    # محاولة تحويل التاريخ
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        profile['datetime'].append(col)
                    except:
                        profile['text'].append(col)
            
            # التحقق من الأعمدة المنطقية
            elif pd.api.types.is_bool_dtype(df[col]):
                profile['boolean'].append(col)
            
        except:
            # في حالة الخطأ، نعتبره نصاً
            profile['text'].append(col)
    
    return profile

def advanced_data_cleaning(df):
    """تنظيف متقدم للبيانات"""
    
    df_clean = df.copy()
    cleaning_log = []
    
    # 1. إزالة الصفوف المكررة بالكامل
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    if len(df_clean) < initial_rows:
        cleaning_log.append(f"✅ تم إزالة {initial_rows - len(df_clean)} صف مكرر")
    
    # 2. إزالة الأعمدة الفارغة تماماً
    empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if empty_cols:
        df_clean.drop(columns=empty_cols, inplace=True)
        cleaning_log.append(f"✅ تم إزالة {len(empty_cols)} عمود فارغ")
    
    # 3. معالجة القيم المفقودة
    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # للأعمدة الرقمية: تعبئة بالمتوسط
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                cleaning_log.append(f"✅ العمود {col}: تم ملء {missing} قيمة مفقودة بالمتوسط")
            else:
                # للأعمدة النصية: تعبئة بالقيمة الأكثر تكراراً
                if not df_clean[col].mode().empty:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    cleaning_log.append(f"✅ العمود {col}: تم ملء {missing} قيمة مفقودة بالقيمة الأكثر تكراراً")
    
    # 4. تنظيف النصوص
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        # إزالة المسافات الزائدة
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # محاولة تحويل الأرقام المخزنة كنص
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
        except:
            pass
    
    # 5. إزالة القيم المتطرفة (للأعمدة الرقمية)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        if len(outliers) > 0:
            cleaning_log.append(f"⚠️ العمود {col}: تم اكتشاف {len(outliers)} قيمة متطرفة (تم الاحتفاظ بها)")
    
    # 6. توحيد حالة النصوص للأعمدة التصنيفية
    categorical_cols = auto_detect_column_types(df_clean)['categorical']
    for col in categorical_cols:
        if col in df_clean.columns and pd.api.types.is_string_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].str.lower().str.strip()
    
    return df_clean, cleaning_log

def detect_data_quality(df):
    """تحليل جودة البيانات بشكل شامل"""
    
    profile = auto_detect_column_types(df)
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'missing_cells_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicates': int(df.duplicated().sum()),
        'duplicates_pct': (df.duplicated().sum() / len(df)) * 100,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'data_types': df.dtypes.value_counts().to_dict(),
        'profile': profile,
        'columns_info': {}
    }
    
    # تحليل مفصل لكل عمود
    for col in df.columns:
        col_info = {
            'type': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique': int(df[col].nunique()),
            'unique_pct': (df[col].nunique() / len(df)) * 100
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'skew': float(df[col].skew()) if not pd.isna(df[col].skew()) else None
            })
        
        report['columns_info'][col] = col_info
    
    return report

def auto_detect_target_column(df):
    """اكتشاف عمود الهدف تلقائياً للتدريب"""
    
    possible_targets = []
    
    # 1. البحث عن أسماء شائعة للأهداف
    target_names = ['target', 'label', 'class', 'fraud', 'corruption', 'risk', 
                    'churn', 'default', 'outlier', 'anomaly', 'y', 'result',
                    'goal', 'output', 'prediction', 'actual', 'status']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(target in col_lower for target in target_names):
            possible_targets.append(col)
    
    # 2. إذا لم نجد، نبحث عن أعمدة بقيم فريدة قليلة (تصنيفية)
    if not possible_targets:
        for col in df.columns:
            if df[col].nunique() <= 10 and df[col].nunique() >= 2:
                possible_targets.append(col)
    
    # 3. إذا لم نجد، نبحث عن أعمدة منطقية
    if not possible_targets:
        for col in df.columns:
            if df[col].dtype == 'bool' or set(df[col].dropna().unique()) <= {0, 1, '0', '1', True, False}:
                possible_targets.append(col)
    
    return possible_targets

# ==================== دوال كشف الشذوذ والفساد ====================

def detect_anomalies_auto(df, contamination=0.1):
    """كشف الشذوذ التلقائي"""
    
    # اختيار الأعمدة الرقمية فقط
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None, "لا توجد أعمدة رقمية كافية للتحليل"
    
    # تجهيز البيانات
    X = df[numeric_cols].fillna(0)
    
    # توحيد المقاييس
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    iso_pred = iso_forest.fit_predict(X_scaled)
    
    # LOF
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20
    )
    lof_pred = lof.fit_predict(X_scaled)
    
    # دمج النتائج
    results = df.copy()
    results['anomaly_score_iso'] = (iso_pred == -1).astype(int)
    results['anomaly_score_lof'] = (lof_pred == -1).astype(int)
    results['anomaly_score'] = (results['anomaly_score_iso'] + results['anomaly_score_lof']) / 2
    results['is_anomaly'] = results['anomaly_score'] > 0.5
    
    return results, numeric_cols

def detect_fraud_patterns_general(df):
    """كشف أنماط عامة للفساد"""
    
    fraud_report = {
        'total_cases': len(df),
        'suspicious_cases': 0,
        'fraud_indicators': [],
        'high_risk_records': [],
        'corruption_score': 0,
        'patterns': []
    }
    
    indicators = []
    
    # 1. تحليل القيم المتطرفة في الأعمدة الرقمية
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # نأخذ أول 5 أعمدة فقط
        if df[col].nunique() > 10:
            mean_val = df[col].mean()
            std_val = df[col].std()
            threshold = mean_val + 3 * std_val
            
            outliers = df[df[col] > threshold]
            if len(outliers) > 0:
                indicators.append({
                    'type': 'numerical_outlier',
                    'column': col,
                    'count': len(outliers),
                    'description': f'قيم متطرفة في عمود {col}'
                })
    
    # 2. تحليل التكرارات في الأعمدة التصنيفية
    cat_cols = auto_detect_column_types(df)['categorical']
    for col in cat_cols[:3]:
        if col in df.columns:
            value_counts = df[col].value_counts()
            most_frequent = value_counts.head(1)
            if len(most_frequent) > 0:
                freq_ratio = most_frequent.values[0] / len(df)
                if freq_ratio > 0.8:  # قيمة واحدة تمثل أكثر من 80%
                    indicators.append({
                        'type': 'high_frequency',
                        'column': col,
                        'count': int(most_frequent.values[0]),
                        'description': f'قيمة "{most_frequent.index[0]}" تمثل {freq_ratio*100:.1f}% في عمود {col}'
                    })
    
    # 3. تحليل العلاقات المشبوهة (إذا وجد أعمدة مناسبة)
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                try:
                    correlation = df[col1].corr(df[col2])
                    if abs(correlation) > 0.95:  # ارتباط قوي جداً
                        indicators.append({
                            'type': 'high_correlation',
                            'columns': f'{col1} و {col2}',
                            'correlation': correlation,
                            'description': f'ارتباط قوي جداً بين {col1} و {col2}'
                        })
                except:
                    pass
    
    fraud_report['fraud_indicators'] = indicators
    fraud_report['suspicious_cases'] = sum(ind.get('count', 0) for ind in indicators)
    if len(df) > 0:
        fraud_report['corruption_score'] = min(fraud_report['suspicious_cases'] / len(df) * 100, 100)
    
    return fraud_report

# ==================== دوال التنبؤ التلقائي ====================

def auto_train_model(df, target_col=None):
    """تدريب نموذج تلقائي على البيانات"""
    
    # اكتشاف الأعمدة المناسبة
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col is None or target_col not in df.columns:
        # محاولة اكتشاف الهدف تلقائياً
        possible_targets = auto_detect_target_column(df)
        if possible_targets:
            target_col = possible_targets[0]
        else:
            return None, "لم يتم العثور على عمود هدف مناسب"
    
    # إزالة الهدف من الميزات
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    if len(feature_cols) < 2:
        return None, "لا توجد ميزات كافية للتدريب"
    
    # تجهيز البيانات
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # تحويل الهدف إلى قيم ثنائية إذا كان تصنيفياً
    if y.dtype == 'object' or y.dtype == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # اختيار النموذج المناسب
    if len(np.unique(y)) == 2:  # تصنيف ثنائي
        if XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
    else:  # تصنيف متعدد أو انحدار
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    # تدريب النموذج
    model.fit(X_train, y_train)
    
    # تقييم
    y_pred = model.predict(X_test)
    
    if len(np.unique(y)) == 2:  # للتصنيف الثنائي
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
    else:  # للتصنيف المتعدد
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    
    # أهمية الميزات
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    result = {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return result, None

# ==================== دوال التصور ====================

def create_data_profile_charts(df, profile):
    """إنشاء رسوم بيانية لملف البيانات"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('توزيع أنواع الأعمدة', 'القيم المفقودة', 'توزيع البيانات', 'جودة البيانات'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'indicator'}]]
    )
    
    # 1. توزيع أنواع الأعمدة
    col_types = {
        'رقمي': len(profile['profile']['numeric']),
        'تصنيفي': len(profile['profile']['categorical']),
        'نصي': len(profile['profile']['text']),
        'تاريخ': len(profile['profile']['datetime']),
        'معرفات': len(profile['profile']['id_columns'])
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(col_types.keys()),
            values=list(col_types.values()),
            marker=dict(colors=['#00ff88', '#ffaa00', '#00ccff', '#ff66aa', '#aa66ff']),
            textinfo='label+value'
        ),
        row=1, col=1
    )
    
    # 2. القيم المفقودة (أعلى 10 أعمدة)
    missing_data = []
    for col, info in profile['columns_info'].items():
        if info['missing'] > 0:
            missing_data.append({
                'column': col[:20],
                'missing': info['missing']
            })
    
    missing_df = pd.DataFrame(missing_data).sort_values('missing', ascending=False).head(10)
    
    if not missing_df.empty:
        fig.add_trace(
            go.Bar(
                x=missing_df['missing'],
                y=missing_df['column'],
                orientation='h',
                marker_color='#ff4b4b'
            ),
            row=1, col=2
        )
    
    # 3. جودة البيانات
    quality_score = 100 - (profile['missing_cells_pct'] + profile['duplicates_pct'] * 2)
    quality_score = max(0, min(100, quality_score))
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=quality_score,
            title={'text': "نقاط جودة البيانات", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [0, 30], 'color': "#ff4b4b"},
                    {'range': [30, 70], 'color': "#ffaa00"},
                    {'range': [70, 100], 'color': "#00ff88"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': quality_score
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="تحليل جودة البيانات",
        title_font_size=20
    )
    
    return fig

# ==================== دوال واجهة المستخدم ====================

def display_header():
    """عرض الهيدر الرئيسي"""
    st.markdown("""
    <div class="main-header">
        <h1>🧹 AI AUTO DATA CLEANER & AUDITOR</h1>
        <p>نظام شامل لتنظيف وتحليل أي بيانات تلقائياً - يدعم CSV, Excel, JSON, TXT</p>
        <div style="margin-top: 2rem;">
            <span class="badge badge-primary">🧹 تنظيف تلقائي</span>
            <span class="badge badge-info">🔍 كشف الشذوذ</span>
            <span class="badge badge-warning">🤖 AutoML</span>
            <span class="badge badge-danger">📊 تحليل شامل</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics_card(title, value, subtitle, color='primary'):
    """عرض بطاقة مقاييس"""
    st.markdown(f"""
    <div class="metric-neon">
        <div class="metric-neon-value">{value}</div>
        <div class="metric-neon-label">{title}</div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.5);">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def display_alert(message, type='info'):
    """عرض تنبيه"""
    alert_class = f"alert-{type}"
    st.markdown(f"""
    <div class="alert {alert_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)

# ==================== الصفحة الرئيسية ====================

def main():
    # تحميل نموذج NLP
    if st.session_state.nlp_model is None and TRANSFORMERS_AVAILABLE:
        with st.spinner("جاري تحميل نموذج الذكاء الاصطناعي..."):
            st.session_state.nlp_model = load_nlp_model()
    
    # عرض الهيدر
    display_header()
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #00ff88;">🔧 لوحة التحكم</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # رفع الملفات
        st.markdown("### 📁 رفع البيانات")
        st.markdown("يدعم: CSV, Excel, JSON, TXT")
        
        uploaded_file = st.file_uploader(
            "اختر ملف",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            key='file_uploader'
        )
        
        if uploaded_file is not None:
            if st.button("🚀 تحميل وتحليل البيانات", use_container_width=True):
                with st.spinner("جاري تحميل وتحليل البيانات..."):
                    try:
                        # تحميل الملف
                        df, error = load_any_file(uploaded_file)
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state.original_df = df
                            
                            # تنظيف البيانات
                            df_clean, cleaning_log = advanced_data_cleaning(df)
                            st.session_state.cleaned_df = df_clean
                            st.session_state.cleaning_report = cleaning_log
                            
                            # تحليل جودة البيانات
                            quality_report = detect_data_quality(df_clean)
                            st.session_state.data_profile = quality_report
                            
                            # اكتشاف الأهداف المحتملة
                            possible_targets = auto_detect_target_column(df_clean)
                            if possible_targets:
                                st.session_state.auto_target = possible_targets[0]
                            
                            st.session_state.data_loaded = True
                            st.session_state.file_info = {
                                'name': uploaded_file.name,
                                'size': f"{uploaded_file.size / 1024:.2f} KB",
                                'rows': len(df),
                                'columns': len(df.columns),
                                'cleaned_rows': len(df_clean),
                                'cleaned_columns': len(df_clean.columns)
                            }
                            
                            st.success(f"✅ تم تحميل {len(df)} سجل وتنظيف البيانات بنجاح")
                    except Exception as e:
                        st.error(f"خطأ في المعالجة: {str(e)}")
        
        st.markdown("---")
        
        # إعدادات التحليل
        if st.session_state.data_loaded:
            st.markdown("### ⚙️ إعدادات التحليل")
            
            contamination = st.slider(
                "حساسية كشف الشذوذ",
                min_value=0.01,
                max_value=0.3,
                value=0.05,
                step=0.01,
                help="نسبة الحالات المتوقعة كشاذة"
            )
            
            if st.button("🔍 كشف الشذوذ", use_container_width=True):
                with st.spinner("جاري تحليل البيانات..."):
                    anomalies_df, features = detect_anomalies_auto(
                        st.session_state.cleaned_df,
                        contamination=contamination
                    )
                    
                    if anomalies_df is not None:
                        st.session_state.anomalies = anomalies_df
                        
                        # تحليل أنماط الفساد
                        fraud_report = detect_fraud_patterns_general(anomalies_df)
                        st.session_state.fraud_report = fraud_report
                        
                        st.success(f"✅ تم اكتشاف {anomalies_df['is_anomaly'].sum()} حالة شاذة")
            
            # اختيار الهدف للتدريب
            possible_targets = auto_detect_target_column(st.session_state.cleaned_df)
            if possible_targets:
                selected_target = st.selectbox(
                    "🎯 اختر عمود الهدف للتدريب",
                    possible_targets,
                    index=0
                )
                
                if st.button("🤖 تدريب نموذج AutoML", use_container_width=True):
                    with st.spinner("جاري تدريب النموذج..."):
                        model_result, error = auto_train_model(
                            st.session_state.cleaned_df,
                            target_col=selected_target
                        )
                        
                        if model_result is not None:
                            st.session_state.model_pack = model_result
                            st.success(f"✅ تم تدريب النموذج بدقة: {model_result['metrics']['accuracy']*100:.1f}%")
                        else:
                            st.warning(f"⚠️ {error}")
        
        st.markdown("---")
        
        # معلومات الملف
        if st.session_state.file_info:
            st.markdown("### 📊 معلومات الملف")
            info = st.session_state.file_info
            st.markdown(f"""
            <div style="background: rgba(0,255,136,0.05); padding: 1rem; border-radius: 12px;">
                <p><strong>الاسم:</strong> {info['name']}</p>
                <p><strong>الحجم:</strong> {info['size']}</p>
                <p><strong>قبل التنظيف:</strong> {info['rows']:,} سجل</p>
                <p><strong>بعد التنظيف:</strong> {info['cleaned_rows']:,} سجل</p>
                <p><strong>الأعمدة:</strong> {info['columns']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        # شاشة الترحيب
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card float-animation">
                <div style="font-size: 3rem; text-align: center;">🧹</div>
                <h3 style="color: #00ff88; text-align: center;">تنظيف تلقائي</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">إزالة التكرارات، معالجة القيم المفقودة، توحيد النصوص</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.2s;">
                <div style="font-size: 3rem; text-align: center;">🔍</div>
                <h3 style="color: #00ff88; text-align: center;">كشف الشذوذ</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">اكتشاف الأنماط غير العادية والقيم المتطرفة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.4s;">
                <div style="font-size: 3rem; text-align: center;">🤖</div>
                <h3 style="color: #00ff88; text-align: center;">AutoML</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">تدريب نماذج تلقائي على أي بيانات</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # إنشاء التبويبات
    tabs = st.tabs([
        "📊 نظرة عامة",
        "🧹 تقرير التنظيف",
        "🔍 كشف الشذوذ",
        "🤖 AutoML",
        "📈 تحليل متقدم"
    ])
    
    # ========== تبويب نظرة عامة ==========
    with tabs[0]:
        if st.session_state.cleaned_df is not None:
            df = st.session_state.cleaned_df
            profile = st.session_state.data_profile
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 نظرة عامة على البيانات</div>', unsafe_allow_html=True)
            
            # مقاييس سريعة
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metrics_card(
                    "إجمالي السجلات",
                    f"{len(df):,}",
                    f"{len(df.columns)} عمود"
                )
            
            with col2:
                numeric_count = len(profile['profile']['numeric'])
                display_metrics_card(
                    "أعمدة رقمية",
                    str(numeric_count),
                    f"{profile['categorical_columns']} تصنيفية"
                )
            
            with col3:
                missing_pct = profile['missing_cells_pct']
                display_metrics_card(
                    "قيم مفقودة",
                    f"{missing_pct:.1f}%",
                    f"{profile['missing_values']} قيمة"
                )
            
            with col4:
                dup_pct = profile['duplicates_pct']
                display_metrics_card(
                    "مكررات",
                    f"{dup_pct:.1f}%",
                    f"{profile['duplicates']} صف"
                )
            
            # رسم بياني لجودة البيانات
            st.markdown("### 📈 تحليل جودة البيانات")
            fig = create_data_profile_charts(df, profile)
            st.plotly_chart(fig, use_container_width=True)
            
            # عرض عينة البيانات
            st.markdown("### 📋 عينة من البيانات (بعد التنظيف)")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تبويب تقرير التنظيف ==========
    with tabs[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧹 تقرير التنظيف التلقائي</div>', unsafe_allow_html=True)
        
        if st.session_state.cleaning_report:
            for log in st.session_state.cleaning_report:
                display_alert(log, type='success' if '✅' in log else 'warning')
        else:
            st.info("لم يتم إجراء أي تغييرات - البيانات نظيفة")
        
        # مقارنة قبل وبعد
        if st.session_state.original_df is not None and st.session_state.cleaned_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 قبل التنظيف")
                st.dataframe(st.session_state.original_df.head(10), use_container_width=True)
            
            with col2:
                st.markdown("### 📊 بعد التنظيف")
                st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تبويب كشف الشذوذ ==========
    with tabs[2]:
        if st.session_state.anomalies is not None:
            anomalies_df = st.session_state.anomalies
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 نتائج كشف الشذوذ</div>', unsafe_allow_html=True)
            
            # مقاييس
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                anomaly_count = anomalies_df['is_anomaly'].sum()
                display_metrics_card(
                    "حالات شاذة",
                    str(anomaly_count),
                    f"{(anomaly_count/len(anomalies_df))*100:.2f}%"
                )
            
            with col2:
                avg_score = anomalies_df['anomaly_score'].mean()
                display_metrics_card(
                    "متوسط درجة الشذوذ",
                    f"{avg_score:.3f}",
                    "0-1"
                )
            
            # عرض الحالات الشاذة
            st.markdown("### 🚫 الحالات الشاذة")
            anomalies_only = anomalies_df[anomalies_df['is_anomaly']]
            st.dataframe(anomalies_only, use_container_width=True)
            
            # تحليل أنماط الفساد
            if st.session_state.fraud_report:
                fraud_report = st.session_state.fraud_report
                
                st.markdown("### 🕵️ مؤشرات الفساد المكتشفة")
                
                if fraud_report['fraud_indicators']:
                    for indicator in fraud_report['fraud_indicators']:
                        display_alert(
                            f"**{indicator['description']}**: {indicator['count']} حالة",
                            type='warning' if indicator['count'] > 10 else 'info'
                        )
                    
                    # درجة الفساد
                    corruption_score = fraud_report['corruption_score']
                    if corruption_score > 50:
                        display_alert(f"⚠️ درجة الفساد عالية: {corruption_score:.1f}%", type='danger')
                    elif corruption_score > 20:
                        display_alert(f"⚠️ درجة الفساد متوسطة: {corruption_score:.1f}%", type='warning')
                    else:
                        display_alert(f"✅ درجة الفساد منخفضة: {corruption_score:.1f}%", type='success')
                else:
                    st.info("لم يتم العثور على مؤشرات فساد واضحة")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 قم بتشغيل كشف الشذوذ من الشريط الجانبي")
    
    # ========== تبويب AutoML ==========
    with tabs[3]:
        if st.session_state.model_pack is not None:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🤖 نتائج AutoML</div>', unsafe_allow_html=True)
            
            st.markdown(f"**الهدف المختار:** {model_pack['target_col']}")
            
            # مقاييس النموذج
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metrics_card(
                    "الدقة",
                    f"{model_pack['metrics']['accuracy']*100:.1f}%",
                    "Accuracy"
                )
            
            with col2:
                display_metrics_card(
                    "Precision",
                    f"{model_pack['metrics']['precision']*100:.1f}%",
                    "دقة التنبؤ"
                )
            
            with col3:
                display_metrics_card(
                    "Recall",
                    f"{model_pack['metrics']['recall']*100:.1f}%",
                    "تغطية الحالات"
                )
            
            with col4:
                display_metrics_card(
                    "F1 Score",
                    f"{model_pack['metrics']['f1']*100:.1f}%",
                    "متوسط الوزن"
                )
            
            # أهمية الميزات
            st.markdown("### 📊 أهمية المتغيرات")
            fig = px.bar(
                model_pack['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='أهم 10 متغيرات',
                color='importance',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 قم بتدريب نموذج AutoML من الشريط الجانبي")
    
    # ========== تبويب تحليل متقدم ==========
    with tabs[4]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 تحليل متقدم</div>', unsafe_allow_html=True)
        
        if st.session_state.cleaned_df is not None:
            # اختيار نوع التحليل
            analysis_type = st.selectbox(
                "نوع التحليل",
                ["إحصائيات وصفية", "ارتباطات", "توزيعات", "تقرير شامل"]
            )
            
            if analysis_type == "إحصائيات وصفية":
                st.markdown("### 📊 إحصائيات وصفية")
                st.dataframe(
                    st.session_state.cleaned_df.describe(include='all'),
                    use_container_width=True
                )
            
            elif analysis_type == "ارتباطات":
                numeric_df = st.session_state.cleaned_df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='Viridis',
                        zmin=-1, zmax=1,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}'
                    ))
                    
                    fig.update_layout(
                        title='مصفوفة الارتباط',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("لا توجد أعمدة رقمية كافية")
            
            elif analysis_type == "توزيعات":
                # اختيار عمود
                numeric_cols = st.session_state.cleaned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("اختر عمود", numeric_cols)
                    
                    fig = px.histogram(
                        st.session_state.cleaned_df,
                        x=selected_col,
                        title=f'توزيع {selected_col}',
                        color_discrete_sequence=['#00ff88']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("لا توجد أعمدة رقمية")
            
            elif analysis_type == "تقرير شامل":
                if st.button("إنشاء تقرير شامل", use_container_width=True):
                    with st.spinner("جاري إنشاء التقرير..."):
                        quality = st.session_state.data_profile
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 ملخص البيانات")
                            st.json({
                                'السجلات': quality['total_rows'],
                                'الأعمدة': quality['total_columns'],
                                'قيم مفقودة': f"{quality['missing_cells_pct']:.1f}%",
                                'مكررات': f"{quality['duplicates_pct']:.1f}%"
                            })
                        
                        with col2:
                            st.markdown("#### 🚨 مؤشرات الجودة")
                            quality_score = 100 - (quality['missing_cells_pct'] + quality['duplicates_pct'] * 2)
                            quality_score = max(0, min(100, quality_score))
                            
                            if quality_score > 80:
                                st.success(f"جودة البيانات: {quality_score:.1f}% (ممتازة)")
                            elif quality_score > 50:
                                st.warning(f"جودة البيانات: {quality_score:.1f}% (متوسطة)")
                            else:
                                st.error(f"جودة البيانات: {quality_score:.1f}% (ضعيفة)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الفوتر
    st.markdown("""
    <div class="footer">
        <h3>🧹 AI Auto Data Cleaner & Auditor</h3>
        <p>الإصدار 3.0 - يدعم جميع أنواع البيانات | جميع الحقوق محفوظة © 2026</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">
            نظام شامل لتنظيف وتحليل أي بيانات تلقائياً باستخدام أحدث تقنيات الذكاء الاصطناعي
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

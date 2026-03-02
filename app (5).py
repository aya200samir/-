import streamlit as st
import pandas as pd
import re
import io
import fitz  # PyMuPDF
import docx
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

# ==========================================
# 1. إعدادات صفحة Streamlit
# ==========================================
st.set_page_config(page_title="المختبر القانوني الذكي - UK ICO", layout="wide")
st.title("⚖️ النظام الخبير لتحليل غرامات البيانات (UK GDPR)")
st.markdown("ارفعي ملفات العقوبات (PDF/Word) ليقوم النظام بتحليلها، تجميعها، واستخراج أسباب الغرامات تلقائياً.")

# ==========================================
# 2. تحميل موديل Legal-BERT (مع ميزة التخزين المؤقت Cache)
# ==========================================
# نستخدم cache_resource لكي لا يتم تحميل الموديل الثقيل مع كل ضغطة زر
@st.cache_resource
def load_legal_bert():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_legal_bert()

# ==========================================
# 3. دوال المعالجة (استخراج النص، المتجهات، والميزات)
# ==========================================
def extract_text_from_file(uploaded_file):
    """استخراج النص من ملفات PDF و Word المرفوعة"""
    text = ""
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.pdf'):
        # قراءة الـ PDF من الذاكرة مباشرة
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
    elif file_name.endswith('.docx'):
        # قراءة الـ Word من الذاكرة
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = "\n".join([p.text for p in doc.paragraphs])
    
    return text

def get_legal_embedding(text):
    """تحويل النص إلى أرقام (Vector) باستخدام Legal-BERT"""
    # نأخذ أول 2000 حرف ونقتطعهم لـ 512 توكن (الحد الأقصى للموديل)
    inputs = tokenizer(text[:2000], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_fine_amount(text):
    """استخراج قيمة الغرامة بالجنيه الإسترليني"""
    match = re.search(r'£(\d{1,3}(?:,\d{3})*)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return 0

# ==========================================
# 4. واجهة المستخدم (رفع الملفات)
# ==========================================
uploaded_files = st.file_uploader("📂 ارفعي ملفات القضايا (PDF, DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 بدء التحليل الذكي (Clustering & Feature Extraction)"):
        with st.spinner('جاري قراءة الملفات وتحليل النصوص القانونية... يرجى الانتظار'):
            
            data = []
            vectors = []
            
            # أ. قراءة النصوص وتحويلها لمتجهات
            for file in uploaded_files:
                text_content = extract_text_from_file(file)
                if text_content.strip():
                    vector = get_legal_embedding(text_content)
                    data.append({"filename": file.name, "text": text_content})
                    vectors.append(vector)
            
            if len(data) > 0:
                # ب. التجميع الذكي (Unsupervised Learning)
                # نحدد عدد المجموعات بناءً على عدد الملفات المرفوعة (حتى لا يحدث خطأ إذا رفعتِ ملفين فقط)
                num_clusters = min(3, len(data)) 
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(vectors)
                
                # ج. استخراج الميزات (Feature Engineering)
                rows = []
                for i, doc in enumerate(data):
                    text = doc['text']
                    fine = extract_fine_amount(text)
                    has_mfa = 1 if "MFA" in text.upper() or "multi-factor authentication" in text.lower() else 0
                    is_children = 1 if "children" in text.lower() else 0
                    has_ransomware = 1 if "ransomware" in text.lower() else 0
                    
                    rows.append({
                        "اسم الملف": doc['filename'],
                        "رقم المجموعة (Cluster)": clusters[i],
                        "قيمة الغرامة (£)": fine,
                        "ثغرة MFA": "نعم" if has_mfa else "لا",
                        "بيانات أطفال": "نعم" if is_children else "لا",
                        "هجوم فدية (Ransomware)": "نعم" if has_ransomware else "لا"
                    })
                
                # تحويل النتائج لجدول
                df_final = pd.DataFrame(rows)
                
                # ==========================================
                # 5. عرض النتائج للسوبر فايزر
                # ==========================================
                st.success("✅ اكتمل التحليل بنجاح!")
                
                st.subheader("📊 جدول المراجعة البشرية (Human-in-the-loop)")
                st.markdown("هذا الجدول يجمع بين ذكاء الآلة في التجميع (Clusters) وبين القواعد لاستخراج الميزات. راجعي النتائج:")
                
                # عرض الجدول بشكل تفاعلي
                st.dataframe(df_final, use_container_width=True)
                
                # زر لتحميل البيانات كـ CSV لتدريب XGBoost لاحقاً
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 تحميل البيانات (Dataset) لتدريب XGBoost",
                    data=csv,
                    file_name='uk_gdpr_gold_dataset.csv',
                    mime='text/csv',
                )
                
            else:
                st.error("لم يتم العثور على نصوص في الملفات المرفوعة.")

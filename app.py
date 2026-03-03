from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import docx
import io
import re
import pandas as pd
import numpy as np
import joblib

# 1. تهيئة السيرفر
app = FastAPI(title="Lumina AI Engine", version="1.0")

# 2. إعدادات الـ CORS (هامة جداً للسماح لـ Lovable بالاتصال بهذا السيرفر)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في المستقبل سنضع هنا رابط Lovable الخاص بكِ فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. محاولة تحميل موديل الذكاء الاصطناعي (XGBoost) إن وجد
try:
    ml_model = joblib.load("lumina_xgboost_model.joblib")
    print("✅ AI Model Loaded Successfully")
except:
    ml_model = None
    print("⚠️ AI Model not found. Running in heuristic fallback mode.")

# 4. دوال المعالجة والاستخراج
def extract_text(file_bytes: bytes, filename: str) -> str:
    text = ""
    try:
        if filename.lower().endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = " ".join([page.get_text() for page in doc])
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = " ".join([p.text for p in doc.paragraphs])
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

def extract_compliance_features(text: str) -> dict:
    text_lower = text.lower()
    
    # البحث عن ثغرات باستخدام RegEx كطبقة أولى
    mfa_match = re.search(r'([^.]*(?:fail\w*|lack of|without|compromised)\s+[^.]*(?:multi[- ]factor authentication|mfa)[^.]*\.)', text, re.IGNORECASE)
    mfa_ev = mfa_match.group(1).strip() if mfa_match else ""
    
    child_match = re.search(r'([^.]*\b(?:child|children|minors|under 13)\b[^.]*\.)', text_lower)
    child_ev = child_match.group(1).strip() if child_match else ""
    
    risk_score = 10
    if mfa_ev: risk_score += 40
    if child_ev: risk_score += 50
    
    return {
        "MFA_Violation": 1 if mfa_ev else 0,
        "MFA_Evidence": mfa_ev,
        "Children_Violation": 1 if child_ev else 0,
        "Children_Evidence": child_ev,
        "Risk_Score": risk_score
    }

# 5. واجهة برمجة التطبيقات (The API Endpoint)
@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # قراءة الملف القادم من Lovable
    file_bytes = await file.read()
    text = extract_text(file_bytes, file.filename)
    
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Document text is too short or unreadable.")
        
    # استخراج الميزات
    features = extract_compliance_features(text)
    
    # حساب الغرامة المتوقعة
    predicted_fine = 0
    if ml_model is not None:
        # إذا كان الموديل موجوداً، نستخدمه
        input_df = pd.DataFrame([{
            'Cluster_ID': 1, # افتراضي
            'Risk_Score': features['Risk_Score'],
            'MFA_Violation': features['MFA_Violation'],
            'Children_Violation': features['Children_Violation']
        }])
        predicted_fine = max(0, int(ml_model.predict(input_df)[0]))
    else:
        # محاكاة ذكية للغرامة في حال عدم وجود الموديل (لأغراض العرض Demo)
        predicted_fine = features['Risk_Score'] * 12500 

    # إرسال النتيجة كـ JSON إلى Lovable
    return {
        "filename": file.filename,
        "risk_score": features['Risk_Score'],
        "predicted_fine_gbp": predicted_fine,
        "critical_findings": {
            "mfa_evidence": features['MFA_Evidence'],
            "children_data_evidence": features['Children_Evidence']
        },
        "status": "success"
    }

@app.get("/")
def health_check():
    return {"status": "Lumina AI Engine is Online"}

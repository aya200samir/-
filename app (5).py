import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from textblob import TextBlob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import networkx as nx
import requests
from streamlit_lottie import st_lottie
import warnings
import os
from datetime import datetime
import xgboost as xgb
warnings.filterwarnings('ignore')

# -------------------------------
# 1. إعداد الصفحة والتصميم
# -------------------------------
st.set_page_config(
    page_title="الرقيب القضائي الذكي - AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# دالة لتحميل Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_p8bfn5sw.json")
lottie_clean = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qwyjxnmr.json")

# حقن CSS مخصص
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    * {
        font-family: 'Cairo', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(0,0,0,1) 0%, rgba(20,30,48,1) 90%);
        color: #e0e0e0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .glass-card {
        background: rgba(20, 30, 48, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 242, 254, 0.8);
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }

    .title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8892b0;
        font-size: 20px;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 2. الهيدر
# -------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_anim")
with col2:
    st.markdown("<p class='title'>الرقيب القضائي الذكي - AutoML</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>منصة ذكاء اصطناعي متكاملة لتنظيف البيانات وتحليلها واكتشاف الأنماط الشاذة</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# 3. كلاس لتنظيف البيانات التلقائي
# -------------------------------
class AutoDataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = []
        self.original_shape = df.shape
        
    def clean(self):
        """تشغيل جميع عمليات التنظيف التلقائي"""
        
        # 1. إزالة الأعمدة الفارغة تماماً
        empty_cols = self.df.columns[self.df.isnull().all()].tolist()
        if empty_cols:
            self.df = self.df.drop(columns=empty_cols)
            self.cleaning_report.append(f"✅ تم إزالة {len(empty_cols)} عمود فارغ تماماً")
        
        # 2. إزالة الصفوف المكررة
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            self.cleaning_report.append(f"✅ تم إزالة {duplicates} صف مكرر")
        
        # 3. معالجة القيم المفقودة
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    # للأعمدة الرقمية: نملأ بالمتوسط
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    self.cleaning_report.append(f"✅ العمود {col}: تم ملء {missing} قيمة مفقودة بالمتوسط")
                else:
                    # للأعمدة النصية: نملأ بالقيمة الأكثر تكراراً
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
                    self.cleaning_report.append(f"✅ العمود {col}: تم ملء {missing} قيمة مفقودة بالقيمة الأكثر تكراراً")
        
        # 4. كشف وإزالة القيم المتطرفة (Outliers) للأعمدة الرقمية
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_count += len(outliers)
                # يمكن اختيار إما الحذف أو التحذير فقط - هنا سنحتفظ بها مع تحذير
                self.cleaning_report.append(f"⚠️ العمود {col}: تم اكتشاف {len(outliers)} قيمة متطرفة (محتفظ بها للتحليل)")
        
        # 5. توحيد حالة النصوص (Lowercase) للأعمدة النصية
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            try:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.cleaning_report.append(f"✅ العمود {col}: تم تنظيف النصوص وإزالة المسافات الزائدة")
            except:
                pass
        
        return self.df
    
    def get_report(self):
        report = f"📊 تقرير التنظيف:\n"
        report += f"- الأبعاد الأصلية: {self.original_shape}\n"
        report += f"- الأبعاد بعد التنظيف: {self.df.shape}\n"
        for item in self.cleaning_report:
            report += f"  {item}\n"
        return report

# -------------------------------
# 4. كلاس للتعلم التلقائي
# -------------------------------
class AutoML:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.features = []
        self.target = None
        self.encoders = {}
        
    def prepare_data(self):
        """تحضير البيانات تلقائياً للتدريب"""
        
        # تحديد الأعمدة الرقمية كنمط
        self.features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # إذا كان هناك عمود "is_suspicious" أو "outlier" نستخدمه كـ target
        target_cols = ['is_suspicious', 'outlier', 'label', 'target', 'class']
        for col in target_cols:
            if col in self.df.columns:
                self.target = col
                if col in self.features:
                    self.features.remove(col)
                break
        
        # إذا لم نجد target، نصنع واحد باستخدام Isolation Forest
        if self.target is None and len(self.features) >= 2:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            self.df['auto_target'] = iso_forest.fit_predict(self.df[self.features])
            self.df['auto_target'] = (self.df['auto_target'] == -1).astype(int)
            self.target = 'auto_target'
            self.features = [f for f in self.features if f != 'auto_target']
        
        return len(self.features) > 0 and self.target is not None
    
    def train_xgboost(self):
        """تدريب نموذج XGBoost"""
        
        if not self.prepare_data():
            return None, "لا توجد بيانات كافية للتدريب"
        
        # تجهيز البيانات
        X = self.df[self.features].fillna(0)
        y = self.df[self.target]
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تدريب النموذج
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # تقييم النموذج
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # أهمية الميزات
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # توقع على كل البيانات
        self.df['ml_score'] = self.model.predict_proba(X)[:, 1] * 100
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'predictions': self.df[['ml_score'] + self.features + [self.target]].copy()
        }, None
    
    def get_feature_importance_plot(self, feature_importance):
        """رسم أهمية الميزات"""
        fig = px.bar(feature_importance.head(10), 
                     x='importance', y='feature', 
                     orientation='h',
                     title='أهم 10 متغيرات في النموذج',
                     color='importance',
                     color_continuous_scale='viridis')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                         paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='white'))
        return fig

# -------------------------------
# 5. واجهة المستخدم الرئيسية
# -------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ لوحة التحكم الذكية</h2>", unsafe_allow_html=True)
    
    # رفع الملفات
    uploaded_file = st.file_uploader("📂 ارفع ملف CSV للتدريب", type=["csv"])
    
    if uploaded_file is not None:
        # قراءة البيانات
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ تم رفع الملف بنجاح! {df_raw.shape[0]} صف، {df_raw.shape[1]} عمود")
        
        # خيار التنظيف التلقائي
        st.markdown("---")
        st.markdown("### 🧹 تنظيف البيانات")
        auto_clean = st.checkbox("تشغيل التنظيف التلقائي", value=True)
        
        # خيار التدريب التلقائي
        st.markdown("---")
        st.markdown("### 🤖 التعلم التلقائي")
        auto_train = st.checkbox("تشغيل التدريب التلقائي (XGBoost)", value=True)
        
        # أزرار التشغيل
        st.markdown("---")
        run_button = st.button("🚀 تشغيل المعالجة", type="primary", use_container_width=True)
    else:
        # بيانات افتراضية إذا لم يتم رفع ملف
        st.info("📌 يرجى رفع ملف CSV للبدء")
        
        # إنشاء بيانات افتراضية للعرض
        np.random.seed(42)
        n_samples = 200
        df_raw = pd.DataFrame({
            'case_id': range(1, n_samples+1),
            'judge': np.random.choice(['قاضي أحمد', 'قاضي خالد', 'قاضي سارة', 'قاضي ليلى', 'قاضي محمد'], n_samples),
            'lawyer': np.random.choice(['محامي علي', 'محامي نور', 'محامي عمر', 'محامي هند', 'محامي سامر'], n_samples),
            'case_type': np.random.choice(['جنائي', 'مدني', 'إداري', 'أسرة'], n_samples),
            'duration_days': np.random.gamma(shape=2, scale=30, size=n_samples).astype(int) + 10,
            'amount': np.random.uniform(1000, 100000, n_samples).round(2),
            'evidence_strength': np.random.uniform(0, 10, n_samples).round(1),
            'sentence_severity': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2,0.3,0.3,0.2]),
            'case_text': [f"نص القضية رقم {i} يحتوي على تفاصيل..." for i in range(1, n_samples+1)]
        })
        
        # إضافة بعض القيم المفقودة للتجربة
        df_raw.loc[0:5, 'amount'] = np.nan
        df_raw.loc[10:15, 'evidence_strength'] = np.nan
        
        auto_clean = True
        auto_train = True
        run_button = True

# -------------------------------
# 6. المعالجة الرئيسية
# -------------------------------
if run_button:
    # تقدم العملية
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # خطوة 1: تنظيف البيانات
    if auto_clean:
        status_text.text("🧹 جاري تنظيف البيانات...")
        progress_bar.progress(20)
        
        cleaner = AutoDataCleaner(df_raw)
        df_cleaned = cleaner.clean()
        cleaning_report = cleaner.get_report()
        
        with st.expander("📋 تقرير التنظيف", expanded=True):
            st.text(cleaning_report)
    else:
        df_cleaned = df_raw.copy()
    
    # خطوة 2: عرض البيانات بعد التنظيف
    status_text.text("📊 عرض البيانات...")
    progress_bar.progress(40)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 البيانات بعد التنظيف")
        st.dataframe(df_cleaned.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 إحصائيات سريعة")
        st.write(f"- عدد الصفوف: {df_cleaned.shape[0]}")
        st.write(f"- عدد الأعمدة: {df_cleaned.shape[1]}")
        st.write(f"- الأعمدة الرقمية: {len(df_cleaned.select_dtypes(include=[np.number]).columns)}")
        st.write(f"- الأعمدة النصية: {len(df_cleaned.select_dtypes(include=['object']).columns)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # خطوة 3: التدريب التلقائي
    if auto_train:
        status_text.text("🤖 جاري تدريب نموذج XGBoost...")
        progress_bar.progress(60)
        
        automl = AutoML(df_cleaned)
        results, error = automl.train_xgboost()
        
        if results:
            # عرض نتائج التدريب
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.metric("🎯 دقة النموذج", f"{results['accuracy']:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.metric("📈 عدد الميزات المستخدمة", len(automl.features))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                high_risk = (results['predictions']['ml_score'] > 70).sum()
                st.metric("🚨 حالات عالية الخطورة", high_risk)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # أهمية الميزات
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.plotly_chart(automl.get_feature_importance_plot(results['feature_importance']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # عرض النتائج مع التصنيف
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 نتائج التصنيف (درجة الاشتباه)")
            
            display_cols = ['ml_score'] + automl.features[:5] + [automl.target]
            df_display = results['predictions'][display_cols].sort_values('ml_score', ascending=False)
            
            # تلوين الدرجات
            def color_score(val):
                if val > 70:
                    return 'background-color: #ff4b4b; color: white;'
                elif val > 40:
                    return 'background-color: #ffa500; color: black;'
                else:
                    return 'background-color: #00f2fe; color: black;'
            
            styled_df = df_display.style.map(color_score, subset=['ml_score'])
            st.dataframe(styled_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # تحليل إضافي: توزيع الدرجات
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig = px.histogram(df_display, x='ml_score', nbins=20, 
                              title='توزيع درجات الاشتباه',
                              color_discrete_sequence=['#ff4b4b'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.error(f"❌ فشل التدريب: {error}")
    
    # خطوة 4: تصدير النتائج
    status_text.text("📥 تجهيز التصدير...")
    progress_bar.progress(90)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📥 تصدير النتائج")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 تحميل البيانات بعد التنظيف"):
            csv = df_cleaned.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="اضغط للتحميل",
                data=csv,
                file_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
    
    with col2:
        if auto_train and results:
            if st.button("📥 تحميل نتائج التصنيف"):
                csv = results['predictions'].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="اضغط للتحميل",
                    data=csv,
                    file_name=f'classified_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # اكتمال العملية
    progress_bar.progress(100)
    status_text.text("✅ تمت المعالجة بنجاح!")
    
    # رسالة نجاح مع تأثير
    st.balloons()

else:
    # عرض تعليمات البدء
    st.markdown("<div class='glass-card' style='text-align:center; padding:50px;'>", unsafe_allow_html=True)
    st.markdown("## 👈 ابدأ برفع ملف CSV من القائمة الجانبية")
    st.markdown("### أو استخدم البيانات التجريبية للاختبار")
    st.markdown("---")
    
    if lottie_clean:
        st_lottie(lottie_clean, height=200, key="clean_anim")
    
    st.markdown("""
    ### 🚀 ميزات النظام:
    - **🧹 تنظيف تلقائي**: إزالة التكرارات، معالجة القيم المفقودة، كشف الشذوذ
    - **🤖 تعلم تلقائي**: تدريب نموذج XGBoost لاكتشاف الأنماط
    - **📊 تحليل متقدم**: رسوم بيانية تفاعلية، تحليل شبكات، NLP
    - **📥 تصدير النتائج**: تحميل البيانات بعد المعالجة والتصنيف
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 7. فوتر
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8892b0; padding:20px;'>
    <p>الرقيب القضائي الذكي - AutoML | جميع الحقوق محفوظة © 2025</p>
    <p style='font-size:12px;'>تم التطوير باستخدام Streamlit, XGBoost, Scikit-learn</p>
</div>
""", unsafe_allow_html=True)

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
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx
import requests
from streamlit_lottie import st_lottie
import time
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. إعداد الصفحة والتصميم
# -------------------------------
st.set_page_config(
    page_title="الرقيب القضائي الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# دالة لتحميل Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_judge = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_u4yrau.json")

# حقن CSS مخصص (Glassmorphism + Neon)
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
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* تصميم الكروت الزجاجية */
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

    /* العناوين */
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

    /* مؤشرات KPIs */
    .kpi-value {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
    }
    .kpi-label {
        color: #8892b0;
        font-size: 18px;
    }

    /* جدول مخصص */
    .dataframe {
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 2. الهيدر مع الأنيميشن
# -------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_judge:
        st_lottie(lottie_judge, height=150, key="judge_anim")
with col2:
    st.markdown("<p class='title'>الرقيب القضائي الذكي</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>منصة متكاملة لكشف الأنماط المشبوهة والفساد في الأحكام باستخدام الذكاء الاصطناعي</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# 3. شريط جانبي للتحكم
# -------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ لوحة التحكم</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 ارفع ملف CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### 🎛️ أوزان مؤشرات الاشتباه")
    weight_stat = st.slider("التحليل الإحصائي", 0, 100, 30, 5)
    weight_nlp = st.slider("تحليل النصوص", 0, 100, 40, 5)
    weight_network = st.slider("تحليل الشبكات", 0, 100, 30, 5)
    st.markdown("---")
    st.markdown("#### 🔍 تصفية النتائج")
    min_risk = st.slider("الحد الأدنى لمؤشر الاشتباه", 0, 100, 0)
    st.markdown("---")
    if st.button("🔄 إعادة تعيين"):
        st.caching.clear_cache()
        st.experimental_rerun()

# -------------------------------
# 4. تحميل البيانات (افتراضية أو مرفوعة)
# -------------------------------
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # بيانات افتراضية تحاكي الواقع القضائي
        np.random.seed(42)
        num_cases = 200
        df = pd.DataFrame({
            'case_id': range(1, num_cases+1),
            'judge': np.random.choice(['قاضي أحمد', 'قاضي خالد', 'قاضي سارة', 'قاضي ليلى', 'قاضي محمد'], num_cases),
            'lawyer': np.random.choice(['محامي علي', 'محامي نور', 'محامي عمر', 'محامي هند', 'محامي سامر'], num_cases),
            'case_type': np.random.choice(['جنائي', 'مدني', 'إداري', 'أسرة'], num_cases),
            'duration_days': np.random.gamma(shape=2, scale=30, size=num_cases).astype(int) + 10,
            'sentence_severity': np.random.choice(['براءة', 'غرامة', 'سجن قصير', 'سجن طويل'], num_cases, p=[0.2,0.3,0.3,0.2]),
            'evidence_strength': np.random.uniform(0, 10, num_cases).round(1),
            'verdict': np.random.choice(['first_party_win', 'second_party_win'], num_cases),
            'case_text': [f"وقائع القضية رقم {i} تتعلق بـ... أدلة الإثبات كانت ... الحكم النهائي ..." for i in range(1, num_cases+1)]
        })
        # إدراج بعض الحالات الشاذة (فساد)
        outlier_idx = np.random.choice(num_cases, size=20, replace=False)
        df.loc[outlier_idx, 'evidence_strength'] = np.random.uniform(8, 10, 20)  # أدلة قوية
        df.loc[outlier_idx, 'sentence_severity'] = 'براءة'  # لكن حكم ببراءة
        df.loc[outlier_idx, 'verdict'] = 'first_party_win'  # الطرف الأول فاز (المتهم)
        # إطالة أو تقصير غير طبيعي للمدة
        df.loc[outlier_idx[0:5], 'duration_days'] = np.random.randint(300, 500, 5)
        df.loc[outlier_idx[5:10], 'duration_days'] = np.random.randint(1, 5, 5)
    return df

df = load_data(uploaded_file)

# -------------------------------
# 5. عرض بيانات أولية
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📋 عينة من البيانات")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 6. KPIs
# -------------------------------
total_cases = len(df)
avg_duration = df['duration_days'].mean()
unique_judges = df['judge'].nunique()
unique_lawyers = df['lawyer'].nunique()

cols = st.columns(4)
with cols[0]:
    st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div class='kpi-label'>إجمالي القضايا</div>
            <div class='kpi-value' style='color:#00f2fe;'>{total_cases}</div>
        </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div class='kpi-label'>متوسط المدة (أيام)</div>
            <div class='kpi-value' style='color:#00f2fe;'>{avg_duration:.1f}</div>
        </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div class='kpi-label'>عدد القضاة</div>
            <div class='kpi-value' style='color:#00f2fe;'>{unique_judges}</div>
        </div>
    """, unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div class='kpi-label'>عدد المحامين</div>
            <div class='kpi-value' style='color:#00f2fe;'>{unique_lawyers}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# 7. تحليل البيانات الاستكشافي (EDA)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📊 تحليل البيانات الاستكشافي")
tab1, tab2, tab3 = st.tabs(["📈 توزيع المدة", "⚖️ توزيع الأحكام", "🔢 مصفوفة ارتباط"])

with tab1:
    fig = px.histogram(df, x='duration_days', nbins=30, title='توزيع مدة القضايا', color_discrete_sequence=['#00f2fe'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    severity_counts = df['sentence_severity'].value_counts().reset_index()
    severity_counts.columns = ['الحكم', 'العدد']
    fig = px.bar(severity_counts, x='الحكم', y='العدد', title='توزيع الأحكام', color_discrete_sequence=['#4facfe'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # حساب مصفوفة ارتباط للأعمدة الرقمية
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='blues', title='مصفوفة الارتباط')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("لا توجد أعمدة رقمية كافية لحساب الارتباط.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 8. تحليل النصوص (WordCloud + Sentiment)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 🧠 تحليل النصوص (NLP)")
col_txt1, col_txt2 = st.columns([1, 1])

with col_txt1:
    if st.button("توليد سحابة الكلمات"):
        # تجميع كل النصوص
        all_text = ' '.join(df['case_text'].astype(str).tolist())
        # إعادة تشكيل العربية
        try:
            reshaped_text = arabic_reshaper.reshape(all_text)
            bidi_text = get_display(reshaped_text)
        except:
            bidi_text = all_text  # في حال عدم وجود مكتبات العربية
        wordcloud = WordCloud(width=800, height=400, background_color='rgba(0,0,0,0)', mode='RGBA', colormap='viridis').generate(bidi_text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)

with col_txt2:
    st.markdown("#### 🔍 تحليل المشاعر (نص القضية)")
    sample_text = st.selectbox("اختر قضية لعرض النص", df['case_id'].tolist())
    text = df[df['case_id'] == sample_text]['case_text'].values[0]
    st.write(text[:500] + "...")
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 to 1
    st.metric("مؤشر المشاعر", f"{sentiment:.2f}", delta=None, delta_color="normal")
    if sentiment > 0.1:
        st.success("نص إيجابي")
    elif sentiment < -0.1:
        st.warning("نص سلبي")
    else:
        st.info("نص محايد")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 9. كشف الشذوذ (Isolation Forest, Z-Score)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 🕵️ كشف الشذوذ الإحصائي")
# اختيار الأعمدة الرقمية
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
if num_features:
    # توحيد المقاييس
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_features].fillna(0))
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X_scaled)
    df['outlier_if'] = outliers  # -1 شاذ، 1 طبيعي
    # Z-Score (أي عمود) - مثال على مدة القضية
    if 'duration_days' in df.columns:
        z_scores = np.abs(stats.zscore(df['duration_days'].fillna(0)))
        df['outlier_z'] = (z_scores > 3).astype(int)  # 1 إذا كان شاذاً
    # عرض النتائج
    col_iso, col_z = st.columns(2)
    with col_iso:
        st.markdown("**Isolation Forest**")
        st.write(f"عدد الحالات الشاذة: {(df['outlier_if'] == -1).sum()}")
    with col_z:
        st.markdown("**Z-Score (المدة)**")
        if 'outlier_z' in df.columns:
            st.write(f"عدد الحالات الشاذة: {df['outlier_z'].sum()}")
else:
    st.info("لا توجد أعمدة رقمية لكشف الشذوذ.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 10. تحليل الشبكات (القضاة - المحامين)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 🔗 تحليل شبكة العلاقات (القضاة والمحامين)")
# بناء رسم بياني بسيط
G = nx.Graph()
# إضافة عقد (قضاة ومحامين)
judges = df['judge'].unique().tolist()
lawyers = df['lawyer'].unique().tolist()
G.add_nodes_from(judges, type='judge')
G.add_nodes_from(lawyers, type='lawyer')
# إضافة حواف لكل قضية
for idx, row in df.iterrows():
    G.add_edge(row['judge'], row['lawyer'], case_id=row['case_id'])
# حساب مقاييس المركزية
centrality = nx.degree_centrality(G)
# ترتيب القضاة حسب الأكثر مركزية
judges_cent = {k: centrality[k] for k in judges if k in centrality}
top_judges = sorted(judges_cent.items(), key=lambda x: x[1], reverse=True)[:5]
st.write("**أكثر القضاة اتصالاً (نشاطاً):**")
for j, c in top_judges:
    st.write(f"- {j}: {c:.3f}")
# رسم الشبكة (اختياري)
if st.checkbox("عرض الشبكة"):
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     mode='lines', line=dict(width=0.5, color='#888')))
    node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], 
                            marker=dict(showscale=False, colorscale='Viridis', size=10))
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 11. نظام التسجيل (Scoring System)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### ⚖️ مؤشر الاشتباه متعدد الأبعاد")
# نحتاج إلى توحيد المؤشرات وجمعها حسب الأوزان
# مؤشر إحصائي (مثلاً من Isolation Forest)
if 'outlier_if' in df.columns:
    df['stat_score'] = (df['outlier_if'] == -1).astype(int) * 50  # 50 إذا كان شاذاً
else:
    df['stat_score'] = 0

# مؤشر NLP (من تحليل المشاعر - نستخدم القيمة المطلقة للانفعال)
df['nlp_score'] = df['case_text'].apply(lambda x: abs(TextBlob(str(x)).sentiment.polarity) * 30)

# مؤشر شبكة (مثلاً درجة الوسطية العالية قد تكون مريبة - نأخذ أعلى 10% كشاذة)
if 'judge' in df.columns:
    # نحسب مركزية كل قاض وننسبها للقضية
    judge_cent = df['judge'].map(centrality).fillna(0)
    # نعتبر القضاة في أعلى 10% مركزية لديهم احتمال اشتباه أعلى
    threshold = np.percentile(judge_cent, 90)
    df['network_score'] = (judge_cent > threshold).astype(int) * 30
else:
    df['network_score'] = 0

# حساب المؤشر الكلي (تطبيع للأوزان)
total_weight = weight_stat + weight_nlp + weight_network
if total_weight > 0:
    df['total_score'] = (df['stat_score'] * weight_stat / 100 +
                         df['nlp_score'] * weight_nlp / 100 +
                         df['network_score'] * weight_network / 100)
else:
    df['total_score'] = 0

# عرض توزيع الدرجات
fig = px.histogram(df, x='total_score', nbins=20, title='توزيع مؤشرات الاشتباه', color_discrete_sequence=['#ff4b4b'])
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 12. جدول النتائج مع التصفية
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📋 قضايا محل الاشتباه")
filtered_df = df[df['total_score'] >= min_risk].sort_values('total_score', ascending=False)

# تلوين الخلفية حسب الخطورة
def color_score(val):
    if val > 70:
        return 'background-color: #ff4b4b; color: white;'
    elif val > 40:
        return 'background-color: #ffa500; color: black;'
    else:
        return 'background-color: #00f2fe; color: black;'

styled_df = filtered_df[['case_id', 'judge', 'lawyer', 'duration_days', 'sentence_severity', 'evidence_strength', 'total_score']].style.map(color_score, subset=['total_score'])
st.dataframe(styled_df, use_container_width=True)

# عرض إحصائيات سريعة
st.write(f"**عدد القضايا التي تتجاوز مؤشر {min_risk}:** {len(filtered_df)}")
if len(filtered_df) > 0:
    st.write(f"**أعلى مؤشر:** {filtered_df['total_score'].max():.1f}")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 13. تصدير التقرير (اختياري)
# -------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📥 تصدير النتائج")
if st.button("تحميل النتائج كـ CSV"):
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="اضغط للتحميل", data=csv, file_name='suspicious_cases.csv', mime='text/csv')
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 14. ملاحظات ختامية
# -------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#8892b0;'>تم التطوير بناءً على أفكار متقدمة في الذكاء الاصطناعي وتحليل البيانات القضائية | جميع الحقوق محفوظة © 2025</div>", unsafe_allow_html=True)

import streamlit as st
import joblib
import numpy as np

# 1. ตั้งค่าหัวข้อหน้าเว็บ
st.set_page_config(page_title="Graduate Admission Predictor", layout="centered")
st.title("🎓 Graduate Admission Prediction")
st.write("กรอกข้อมูลของคุณเพื่อพยากรณ์โอกาสในการเข้าเรียนต่อ")

# 2. โหลดโมเดล
@st.cache_resource # ใช้ cache เพื่อให้โหลดโมเดลครั้งเดียว ช่วยให้เว็บเร็วขึ้น
def load_model():
    return joblib.load('admission_model.pkl')

model = load_model()

# 3. สร้างฟอร์มรับข้อมูล (5 Features)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        gre = st.number_input("GRE Score (0-340)", min_value=0, max_value=340, value=300)
        toefl = st.number_input("TOEFL Score (0-120)", min_value=0, max_value=120, value=100)
        univ_rating = st.slider("University Rating", 1, 5, 3)
    with col2:
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, 0.5)
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, 0.5)

# 4. ส่วนการพยากรณ์
if st.button("Predict Probability"):
    # จัดเรียงข้อมูลให้ตรงกับที่ Train (GRE, TOEFL, Univ_Rating, SOP, LOR)
    input_data = np.array([[gre, toefl, univ_rating, sop, lor]])
    prediction = model.predict(input_data)
    
    # แสดงผล
    probability = prediction[0] * 100
    st.divider()
    st.subheader(f"โอกาสในการเข้าเรียนของคุณคือ: {probability:.2f}%")
    
    if probability >= 75:
        st.success("โอกาสสูงมาก! เตรียมตัวยื่นใบสมัครได้เลย")
    elif probability >= 50:
        st.warning("มีลุ้น แนะนำให้เพิ่มผลงานหรือคะแนนในส่วนอื่นๆ")
    else:
        st.error("ค่อนข้างท้าทาย แนะนำให้ปรับปรุงคะแนนสอบเพิ่มเติม")
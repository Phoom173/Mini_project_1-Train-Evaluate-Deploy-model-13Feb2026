import joblib
import numpy as np

# 1. โหลดโมเดลที่เซฟไว้กลับมา
loaded_model = joblib.load('admission_model.pkl')

# 2. จำลองข้อมูลใหม่ (5 Features: GRE, TOEFL, Univ_Rating, SOP, LOR)
# เช่น เพื่อนมีคะแนน GRE: 320, TOEFL: 110, Univ_Rating: 4, SOP: 4.0, LOR: 4.5
new_data = np.array([[320, 110, 4, 4.0, 4.5]])

# 3. สั่งทำนายผล
prediction = loaded_model.predict(new_data)

print(f"โอกาสที่จะได้รับการตอบรับเข้าเรียนคือ: {prediction[0]*100:.2f}%")
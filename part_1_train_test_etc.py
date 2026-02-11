import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Data
df = pd.read_csv('Admission_Predict_Ver1.1.csv')

# 2. Select 5 Features and Target
# หมายเหตุ: ในไฟล์ต้นฉบับ 'LOR ' และ 'Chance of Admit ' มีช่องว่างปิดท้ายชื่อคอลัมน์
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ']
target = 'Chance of Admit '

X = df[features]
y = df[target]

# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Save Model
joblib.dump(model, 'admission_model.pkl')
print("Model saved as 'admission_model.pkl'")
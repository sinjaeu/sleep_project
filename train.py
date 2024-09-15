import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 데이터 로드
se_df = pd.read_csv('sleep_ef.csv')
sh_df = pd.read_csv('sleep_healthy.csv')
ssd_df = pd.read_csv('SleepStudyData.csv')
sq_df = pd.read_csv('sleep_quality.csv')

# 결측치 처리
se_df = se_df.fillna(0)
ssd_df = ssd_df.dropna()

# 매핑 정의
gender_mapping = {
    'Male': 0,
    'Female': 1
}

bmi_mapping = {
    "Normal": 0,
    "Normal Weight": 1,
    "Overweight": 2,
    "Obese": 3
}

disorder_mapping = {
    0: 0,
    'Insomnia': 1,
    'Sleep Apnea': 0
}

answer_mapping = {
    'Yes': 1,
    'No': 0
}

# 데이터 전처리
sh_df['BMI Category'] = sh_df['BMI Category'].map(bmi_mapping)
sh_df['Sleep Disorder'] = sh_df['Sleep Disorder'].map(disorder_mapping)
sh_df['Gender'] = sh_df['Gender'].map(gender_mapping)

ssd_df['Enough'] = ssd_df['Enough'].map(answer_mapping)
ssd_df['PhoneReach'] = ssd_df['PhoneReach'].map(answer_mapping)
ssd_df['PhoneTime'] = ssd_df['PhoneTime'].map(answer_mapping)
ssd_df['Breakfast'] = ssd_df['Breakfast'].map(answer_mapping)

se_df['Gender'] = se_df['Gender'].map(gender_mapping)
se_df['Smoking status'] = se_df['Smoking status'].map(answer_mapping)

# 수면 점수 반올림
sq_df['Sleep_Quality_Score'] = sq_df['Sleep_Quality_Score'].round()
se_df['Sleep duration'] = se_df['Sleep duration'].round()
sh_df['Sleep Duration'] = sh_df['Sleep Duration'].round()

se_df['Sleep_Quality_Score'] = se_df['Sleep duration']
sh_df['Sleep_Quality_Score'] = sh_df['Sleep Duration']

min_tired = ssd_df['Tired'].min()
max_tired = ssd_df['Tired'].max()

min_score = 1.0
max_score = 10.0
b = (max_score - min_score) / (max_tired - min_tired)
a = max_score

ssd_df['Sleep_Quality_Score'] = round(a - b * ssd_df['Tired'])

# 데이터 통합
combined_df = pd.concat([se_df, sh_df, ssd_df, sq_df], ignore_index=True)
# 삭제할 열 정의
columns_to_drop = ['Sleep duration', 'Sleep Duration', 'Tired']

# 존재하는 열만 삭제
existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
combined_df = combined_df.drop(columns=existing_columns_to_drop)

# 특성 및 타겟 정의
X = combined_df.drop('Sleep_Quality_Score', axis=1)  # 타겟 변수를 제외한 특성
y = combined_df['Sleep_Quality_Score']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=2, random_state=42)

# 교차 검증
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validated R-squared: {cv_scores.mean()}")

# 전체 데이터로 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 예측값과 실제값의 비교
plt.figure(figsize=(12, 6))

# 산점도: 실제 값 vs 예측 값
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Sleep Quality Score')
plt.ylabel('Predicted Sleep Quality Score')
plt.title('Actual vs Predicted Sleep Quality Score')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 대각선

# 잔차 분석
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sleep Quality Score')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

plt.tight_layout()
plt.show()

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# 'Sleep_Quality_Score'와의 상관관계만 추출
corr_with_sleep_quality = combined_df.corr()['Sleep_Quality_Score'].sort_values(ascending=False)

# 결과 출력
print(corr_with_sleep_quality)

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_sleep_quality.index, y=corr_with_sleep_quality.values, hue=corr_with_sleep_quality.index, dodge=False, palette='coolwarm', legend=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Correlation with Sleep_Quality_Score')
plt.title('Correlation with Sleep_Quality_Score')
plt.show()
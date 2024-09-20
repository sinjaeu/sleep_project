import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 데이터 로드
se_df = pd.read_csv('data/sleep_ef.csv')
sh_df = pd.read_csv('data/sleep_healthy.csv')
ssd_df = pd.read_csv('data/SleepStudyData.csv')
sq_df = pd.read_csv('data/sleep_quality.csv')

# 결측치 처리
se_df = se_df.fillna(0)
ssd_df = ssd_df.dropna()

# 매핑 정의
gender_mapping = {'Male': 0, 'Female': 1}
bmi_mapping = {"Normal": 0, "Normal Weight": 1, "Overweight": 2, "Obese": 3}
disorder_mapping = {0: 0, 'Insomnia': 1, 'Sleep Apnea': 2}
answer_mapping = {'Yes': 1, 'No': 0}

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

# 수면 점수 반올림 및 변환
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

# Quality of Sleep을 Sleep_Quality_Score로 대체
if 'Quality of Sleep' in sq_df.columns:
    sq_df = sq_df.rename(columns={'Quality of Sleep': 'Sleep_Quality_Score'})

# 데이터 통합
combined_df = pd.concat([se_df, sh_df, ssd_df, sq_df], ignore_index=True)

# 중복된 열 제거
columns_to_drop = ['Sleep duration', 'Sleep Duration', 'Tired', 'Quality of Sleep']
existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
combined_df = combined_df.drop(columns=existing_columns_to_drop)

# 특성 및 타겟 정의
X = combined_df.drop('Sleep_Quality_Score', axis=1)  # 타겟 변수를 제외한 특성
y = combined_df['Sleep_Quality_Score']

# 'Sleep_Quality_Score'와의 상관관계만 추출
corr_with_sleep_quality = combined_df.corr()['Sleep_Quality_Score'].sort_values(ascending=False)

# 상관계수 절대값 기준 설정
corr_threshold = 0.1

# 상관계수 절대값이 기준 이하인 열 필터링
low_corr_columns = corr_with_sleep_quality[abs(corr_with_sleep_quality) < corr_threshold].index

# 필터링된 피처로 데이터프레임 생성
X_filtered = X.drop(columns=low_corr_columns)
print(X_filtered.columns)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# 필터링된 피처에 대한 특성 스케일링
scaler_filtered = StandardScaler()
X_train_filtered = scaler_filtered.fit_transform(X_train)
X_test_filtered = scaler_filtered.transform(X_test)

# 필터링된 피처로 모델 학습
model_filtered = RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=2, random_state=42)
model_filtered.fit(X_train_filtered, y_train)

# 예측
y_pred_filtered = model_filtered.predict(X_test_filtered)

# 평가
mse_filtered = mean_squared_error(y_test, y_pred_filtered)
r2_filtered = r2_score(y_test, y_pred_filtered)
print(f"Mean Squared Error (Filtered Features): {mse_filtered}")
print(f"R-squared (Filtered Features): {r2_filtered}")
print(f"Training Score: {model_filtered.score(X_train_filtered, y_train)}")
print(f"Testing Score: {model_filtered.score(X_test_filtered, y_test)}")

# 모델 저장
feature_names = X_filtered.columns.tolist()
joblib.dump(model_filtered, 'model/model_filtered.pkl')
joblib.dump(scaler_filtered, 'model/scaler_filtered.pkl')
joblib.dump(feature_names, 'model/feature_names.pkl')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
se_df = pd.read_csv('sleep_ef.csv')
sh_df = pd.read_csv('sleep_healthy.csv')
ssd_df = pd.read_csv('SleepStudyData.csv')
sq_df = pd.read_csv('sleep_quality.csv')

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

print(combined_df.info())
# 중복된 열 제거
columns_to_drop = ['Sleep duration', 'Sleep Duration', 'Tired', 'Quality of Sleep']
existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
combined_df = combined_df.drop(columns=existing_columns_to_drop)

# 특성 및 타겟 정의
X = combined_df.drop('Sleep_Quality_Score', axis=1)  # 타겟 변수를 제외한 특성
y = combined_df['Sleep_Quality_Score']

# 'Sleep_Quality_Score'와의 상관관계만 추출
corr_with_sleep_quality = combined_df.corr()['Sleep_Quality_Score'].sort_values(ascending=False)

# 상관계수 절대값 기준 설정 (예: 0.1 이하인 열 제외)
corr_threshold = 0.1

# 상관계수 절대값이 기준 이하인 열 필터링
low_corr_columns = corr_with_sleep_quality[abs(corr_with_sleep_quality) < corr_threshold].index

# 필터링된 피처로 데이터프레임 생성
X_filtered = X.drop(columns=low_corr_columns)

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

# 매핑 정보
mappings = {
    'Gender': gender_mapping,
    'BMI Category': bmi_mapping,
    'Sleep Disorder': disorder_mapping,
    'Enough': answer_mapping,
    'PhoneReach': answer_mapping,
    'PhoneTime': answer_mapping,
    'Breakfast': answer_mapping,
    'Smoking status': answer_mapping
}

# 필드별 최대값과 최소값
ranges = {
    'Physical Activity Level': {'min': X_filtered['Physical Activity Level'].min(), 'max': X_filtered['Physical Activity Level'].max()},
    'Stress Level': {'min': X_filtered['Stress Level'].min(), 'max': X_filtered['Stress Level'].max()},
    'Heart Rate': {'min': X_filtered['Heart Rate'].min(), 'max': X_filtered['Heart Rate'].max()},
    'Hours': {'min': X_filtered['Hours'].min(), 'max': X_filtered['Hours'].max()},
    'Caffeine_Intake_mg': {'min': X_filtered['Caffeine_Intake_mg'].min(), 'max': X_filtered['Caffeine_Intake_mg'].max()}
}

# 사용자 입력 데이터 받기
print("현재 입력값을 입력해 주세요:")
input_data_dict = {}
for column in X_filtered.columns:
    if column in mappings:
        print(f"{column}의 값의 범위 및 매핑: {mappings[column]}")
        while True:
            try:
                value = int(input(f"{column}의 값을 입력하세요: "))
                if value in mappings[column].values():
                    input_data_dict[column] = value
                    break
                else:
                    print(f"입력값 {value}가 유효하지 않습니다. 다시 입력하세요. ({column}의 유효한 값: {list(mappings[column].keys())})")
            except ValueError:
                print("정수를 입력해 주세요.")
    elif column in ranges:
        min_value = ranges[column]['min']
        max_value = ranges[column]['max']
        print(f"{column}의 범위: {min_value} ~ {max_value}")
        while True:
            try:
                value = float(input(f"{column}의 값을 입력하세요: "))
                if min_value <= value <= max_value:
                    input_data_dict[column] = value
                    break
                else:
                    print(f"입력값 {value}가 범위를 벗어났습니다. {column}의 범위: {min_value} ~ {max_value}")
            except ValueError:
                print("숫자 형식으로 입력해 주세요.")
    else:
        print(f"{column}의 범위: 없음")
        while True:
            try:
                value = float(input(f"{column}의 값을 입력하세요: "))
                input_data_dict[column] = value
                break
            except ValueError:
                print("숫자 형식으로 입력해 주세요.")

# 입력 데이터로 예측
input_data_filtered_df = pd.DataFrame([input_data_dict], columns=X_filtered.columns)
input_data_filtered_scaled = scaler_filtered.transform(input_data_filtered_df)
predicted_sleep_quality_score = model_filtered.predict(input_data_filtered_scaled)

# 예측 결과 조정: 1보다 낮으면 1로 설정
adjusted_sleep_quality_score = max(predicted_sleep_quality_score[0], 1.0)
print(f"현재 입력값으로 예측된 수면 점수: {adjusted_sleep_quality_score:.2f}")

# 피드백 제공 함수
def generate_feedback(input_data):
    feedback = []

    # 카페인 섭취
    caffeine_intake = input_data.get('Caffeine_Intake_mg', 0)
    if caffeine_intake > 150:
        feedback.append("카페인 섭취가 높습니다. 카페인 섭취를 줄이면 수면 질이 개선될 수 있습니다.")
    
    # 스트레스 수준
    stress_level = input_data.get('Stress Level', 0)
    if stress_level > 5:
        feedback.append("스트레스 수준이 높습니다. 스트레스를 줄이기 위한 명상이나 이완 기법을 시도해 보세요.")
    
    # 신체 활동 수준
    physical_activity_level = input_data.get('Physical Activity Level', 0)
    if physical_activity_level < 30:
        feedback.append("신체 활동 수준이 낮습니다. 매일 규칙적인 운동을 통해 신체 활동을 증가시키세요.")
    
    # 수면 장애
    sleep_disorder = input_data.get('Sleep Disorder', 0)
    if sleep_disorder >= 1:
        feedback.append("수면 장애가 있습니다. 전문가의 상담을 통해 적절한 치료를 받는 것이 좋습니다.")
    
    # BMI 카테고리
    bmi_category = input_data.get('BMI Category', 0)
    if bmi_category >= 2:
        feedback.append("체중이 높은 범주에 속합니다. 체중 조절을 통해 수면 질을 개선할 수 있습니다.")
    
    # 충분한 수면
    enough_sleep = input_data.get('Enough', 0)
    if enough_sleep == 0:
        feedback.append("충분한 수면을 취하지 못하고 있습니다. 일관된 수면 일정을 유지하세요.")
    
    # 아침 식사
    breakfast = input_data.get('Breakfast', 0)
    if breakfast == 0:
        feedback.append("아침 식사를 건너뛰는 경향이 있습니다. 규칙적인 아침 식사가 수면 질에 긍정적인 영향을 미칠 수 있습니다.")
    
    # 추가적인 피드백
    if not feedback:
        feedback.append("현재 입력값으로는 추가적인 조정이 필요하지 않을 수 있습니다.")
    
    return feedback

# 피드백 생성
feedback = generate_feedback(input_data_dict)

# 피드백 출력
print("수면 점수를 개선하기 위한 피드백:")
for item in feedback:
    print(f"- {item}")

from flask import Flask, request, render_template, jsonify
import threading
import time
import uuid
import pandas as pd
import joblib

app = Flask(__name__)

task_status = {}

# 모델 불러오기
model_filtered = joblib.load('model/model_filtered.pkl')
scaler_filtered = joblib.load('model/scaler_filtered.pkl')
feature_names = joblib.load('model/feature_names.pkl')

# 메핑
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'BMI Category': {"Normal": 0, "Normal Weight": 1, "Overweight": 2, "Obese": 3},
    'Sleep Disorder': {'No': 0, 'Insomnia': 1, 'Sleep Apnea': 2},
    'Enough': {'Yes': 1, 'No': 0},
    'PhoneReach': {'Yes': 1, 'No': 0},
    'PhoneTime': {'Yes': 1, 'No': 0},
    'Breakfast': {'Yes': 1, 'No': 0},
    'Smoking status': {'Yes': 1, 'No': 0}
}

ranges = {
    'Physical Activity Level': {'min': 0, 'max': 100},
    'Stress Level': {'min': 0, 'max': 10},
    'Heart Rate': {'min': 60, 'max': 180},
    'Hours': {'min': 4, 'max': 12},
    'Caffeine_Intake_mg': {'min': 0, 'max': 500}
}

# 예측, 피드백 함수
def predict_sleep_quality(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0.0
    
    input_df = input_df[feature_names]
    
    # 입력 데이터 스케일링
    input_scaled = scaler_filtered.transform(input_df)
    
    # 점수 예측
    predicted_score = model_filtered.predict(input_scaled)
    return max(predicted_score[0], 1.0)  # 기본 점수 세팅

def generate_feedback(input_data, sleep_quality):
    feedback = []
    if input_data['Caffeine_Intake_mg'] >= 100:
        feedback.append("카페인 섭취가 높습니다. 카페인 섭취를 줄이세요.")
    if input_data['Stress Level'] > 5:
        feedback.append("스트레스가 높습니다. 이완 기법을 시도해보세요.")
    if input_data['Physical Activity Level'] < 30:
        feedback.append("신체 활동이 부족합니다. 운동을 늘리세요.")
    if input_data['Sleep Disorder'] >= 1:
        feedback.append("수면 장애가 있습니다. 전문가와 상담하세요.")
    if input_data['BMI Category'] >= 2:
        feedback.append("체중이 높습니다. 체중 조절을 고려하세요.")
    if input_data['Enough'] == 0:
        feedback.append("충분한 수면을 취하지 못하고 있습니다.")
    if input_data['Breakfast'] == 0:
        feedback.append("아침 식사를 건너뛰고 있습니다.")
    if input_data['Hours'] <= 6:
        feedback.append("수면 시간이 너무 적습니다.")
    if input_data['Hours'] >= 12:
        feedback.append("수면 시간이 너무 많습니다.")
    if sleep_quality < 5:
        feedback.append("수면 점수가 너무 낮습니다.")
    if not feedback:
        feedback.append("현재 상태는 적절합니다.")
    return feedback

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for column in feature_names:
            value = request.form.get(column)
            if value is not None:
                if column in ['Gender', 'BMI Category', 'Sleep Disorder', 'Enough', 'PhoneReach', 'PhoneTime', 'Breakfast', 'Smoking status']:
                    input_data[column] = int(value)
                else:
                    input_data[column] = float(value)
        
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = 0.0

        sleep_quality_score = round(predict_sleep_quality(input_data), 2)
        feedback = generate_feedback(input_data, sleep_quality_score)
        
        return jsonify({
            "sleep_quality_score": sleep_quality_score,
            "feedback": feedback
        })
    except Exception as e:
        # 오류 로깅
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/sleep_health')
def sleep_health():
    return render_template('sleep_health.html')

if __name__ == '__main__':
    app.run(debug=True)

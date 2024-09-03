import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('sleep_quality.csv')

# 독립 변수와 종속 변수 정의
X = df[['Caffeine_Intake_mg', 'Stress_Level', 'Body_Temperature', 'Movement_During_Sleep', 
        'Sleep_Duration_Hours', 'Bedtime_Consistency', 'Light_Exposure_hours']]
y = df['Sleep_Quality_Score']

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gradient Boosting 모델
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_gb_pred = gb_model.predict(X_test_scaled)

# 성능 평가
mse_gb = mean_squared_error(y_test, y_gb_pred)
r2_gb = r2_score(y_test, y_gb_pred)

print(f"Gradient Boosting Mean Squared Error: {mse_gb:.2f}")
print(f"Gradient Boosting R^2 Score: {r2_gb:.2f}")

# Gradient Boosting 하이퍼파라미터 튜닝
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
gb_grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=gb_param_grid, cv=5, scoring='r2')
gb_grid_search.fit(X_train_scaled, y_train)

print(f"Best Gradient Boosting Parameters: {gb_grid_search.best_params_}")
print(f"Best Gradient Boosting R^2 Score: {gb_grid_search.best_score_:.2f}")

# 최적의 Gradient Boosting 모델로 다시 예측
best_gb_model = gb_grid_search.best_estimator_
y_best_gb_pred = best_gb_model.predict(X_test_scaled)

# 최적의 Gradient Boosting 모델 성능 평가
mse_best_gb = mean_squared_error(y_test, y_best_gb_pred)
r2_best_gb = r2_score(y_test, y_best_gb_pred)

print(f"Optimized Gradient Boosting Mean Squared Error: {mse_best_gb:.2f}")
print(f"Optimized Gradient Boosting R^2 Score: {r2_best_gb:.2f}")

# Gradient Boosting 특성 중요도 시각화
feature_importances = best_gb_model.feature_importances_

plt.figure(figsize=(12, 8))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance')
plt.show()

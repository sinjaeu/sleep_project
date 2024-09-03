import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터 로드
df = pd.read_csv('sleep_quality.csv')

# 상관관계 행렬 계산
corr_matrix = df.corr()

# 상관관계 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# 독립 변수와 종속 변수 정의
X = df[['Caffeine_Intake_mg', 'Stress_Level', 'Body_Temperature', 'Movement_During_Sleep', 
        'Sleep_Duration_Hours', 'Bedtime_Consistency', 'Light_Exposure_hours']]
y = df['Sleep_Quality_Score']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 회귀 모델 적합
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# 회귀 분석 결과 출력
print("Linear Regression Model Summary:")
print(model.summary())

# 다항 회귀 모델 설정
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train[['Caffeine_Intake_mg']])
poly_model = LinearRegression().fit(X_poly, y_train)

# 다항 회귀 예측 및 시각화
X_range = np.linspace(df['Caffeine_Intake_mg'].min(), df['Caffeine_Intake_mg'].max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = poly_model.predict(X_range_poly)

plt.figure(figsize=(8, 6))
plt.scatter(df['Caffeine_Intake_mg'], df['Sleep_Quality_Score'], color='blue', label='Data')
plt.plot(X_range, y_pred, color='red', label='Polynomial Fit')
plt.xlabel('Caffeine Intake (mg)')
plt.ylabel('Sleep Quality Score')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

se_df = pd.read_csv('sleep_ef.csv')
sh_df = pd.read_csv('sleep_healthy.csv')
ssd_df = pd.read_csv('SleepStudyData.csv')
sq_df = pd.read_csv('sleep_quality.csv')

se_df = se_df.fillna(0)
ssd_df = ssd_df.dropna()

gender_mapping = {
    'Male' : 0,
    'Female' : 1
}

# 'BMI Category' 열의 범주형 데이터를 숫자형으로 변환
bmi_mapping = {
    "Normal": 0,
    "Normal Weight": 1,
    "Overweight": 2,
    "Obese": 3
}

# 데이터프레임의 BMI 열을 매핑된 숫자로 변환
sh_df['BMI Category'] = sh_df['BMI Category'].map(bmi_mapping)
sh_df['Sleep Disorder'] = sh_df['Sleep Disorder'].fillna(0)

answer_mapping = {
    'Yes' : 1,
    'No' : 0
}

ssd_df['Enough'] = ssd_df['Enough'].map(answer_mapping)
ssd_df['PhoneReach'] = ssd_df['PhoneReach'].map(answer_mapping)
ssd_df['PhoneTime'] = ssd_df['PhoneTime'].map(answer_mapping)
ssd_df['Breakfast'] = ssd_df['Breakfast'].map(answer_mapping)

se_df.info()
sh_df.info()
ssd_df.info()
sq_df.info()
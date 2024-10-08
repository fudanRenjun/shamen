import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载随机森林模型
model = joblib.load('E:/RS/sm/ALL-Feature/XGBoost/final/XGBoost-16.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "10975", "8787", "5491", "8463", "6838", "16044","5374","5619",
    "12361","2996","11642","4940","5680","2844","9864","3857"
]

# Streamlit 用户界
st.title("Predict Salmonella subtypes by integrating MALDI-TOF MS and XGBoost model")

# 用户输入特征数据
input_10975 = st.number_input("10975:", min_value=0.0, max_value=100.0, value=10.0)
input_8787 = st.number_input("8787:", min_value=0.0, max_value=100.0, value=10.0)
input_5491 = st.number_input("5491:", min_value=0.0, max_value=100.0, value=10.0)
input_8463 = st.number_input("8463:", min_value=0.0, max_value=100.0, value=10.0)
input_6838 = st.number_input("6838:", min_value=0.0, max_value=100.0, value=10.0)
input_16044 = st.number_input("16044:", min_value=0.0, max_value=100.0, value=10.0)
input_5374 = st.number_input("5374:", min_value=0.0, max_value=100.0, value=10.0)
input_5619 = st.number_input("5619:", min_value=0.0, max_value=100.0, value=10.0)
input_12361 = st.number_input("12361:", min_value=0.0, max_value=100.0, value=10.0)
input_2996 = st.number_input("2996:", min_value=0.0, max_value=100.0, value=10.0)
input_11642 = st.number_input("11642:", min_value=0.0, max_value=100.0, value=10.0)
input_4940 = st.number_input("4940:", min_value=0.0, max_value=100.0, value=10.0)
input_5680 = st.number_input("5680:", min_value=0.0, max_value=100.0, value=10.0)
input_2844 = st.number_input("2844:", min_value=0.0, max_value=100.0, value=10.0)
input_9864 = st.number_input("9864:", min_value=0.0, max_value=100.0, value=10.0)
input_3857 = st.number_input("3857:", min_value=0.0, max_value=100.0, value=10.0)

# 将输入的数据转化为模型的输入格式
feature_values = [
    input_10975, input_8787, input_5491, input_8463, input_6838, input_16044,
    input_5374, input_5619, input_12361, input_2996, input_11642, input_4940,
    input_5680, input_2844, input_9864, input_3857
]
features = np.array([feature_values])

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0-7: 对应分类)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100
    advice = f"The model predicts that your probability of being in class {predicted_class} is {probability:.1f}%."

    st.write(advice)
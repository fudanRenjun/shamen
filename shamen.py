import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载随机森林模型
model = joblib.load('XGBoost-16.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "10975", "8787", "5491", "8463", "6838", "16044","5374","5619",
    "12361","2996","11642","4940","5680","2844","9864","3857"
]

# Streamlit 用户界
st.title("Predict Salmonella subtypes by integrating MALDI-TOF MS and XGBoost model")
st.write('Please enter the following indicators to Predict:')

# 用户输入特征数据
input_10975 = st.number_input("10975±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_8787 = st.number_input("8787±1:", min_value=0.0000000001, max_value=1.0, value=1.0,format="%.9f")
input_5491 = st.number_input("5491±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_8463 = st.number_input("8463±1:", min_value=0.0000000001, max_value=1.0, value=1.0,format="%.9f")
input_6838 = st.number_input("6838±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_16044 = st.number_input("16044±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_5374 = st.number_input("5374±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_5619 = st.number_input("5619±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_12361 = st.number_input("12361±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_2996 = st.number_input("2996±1:", min_value=0.00000001, max_value=1.0, value=1.0,format="%.9f")
input_11642 = st.number_input("11642±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_4940 = st.number_input("4940±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_5680 = st.number_input("5680±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_2844 = st.number_input("2844±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_9864 = st.number_input("9864±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")
input_3857 = st.number_input("3857±1:", min_value=0.000000001, max_value=1.0, value=1.0,format="%.9f")

# 将输入的数据转化为模型的输入格式
feature_values = [
    input_10975, input_8787, input_5491, input_8463, input_6838, input_16044,
    input_5374, input_5619, input_12361, input_2996, input_11642, input_4940,
    input_5680, input_2844, input_9864, input_3857
]
features = np.array([feature_values])

# 数字标签和文本标签的映射关系
label_mapping = {
    0: "S.Group B",
    1: "S.Group C1",
    2: "S.Group C2/3",
    3: "S.enteritidis subtype",
    4: "S.Group D",
    5: "S.Group E1",
    6: "Non-A-F group",
    7: "S.typhimurium subtype"
}

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    predicted_label = label_mapping[predicted_class]
    st.write(f"**Predicted Class:** {predicted_label}")

    # 显示每个类别的预测概率
    st.write("**Prediction Probabilities:**")
    for i, proba in enumerate(predicted_proba):
        label = label_mapping[i]
        st.write(f"{label}: {proba * 100:.2f}%")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100
    advice = f"The model predicts that your probability of being in class {predicted_label} is {probability:.1f}%."
    st.write(advice)

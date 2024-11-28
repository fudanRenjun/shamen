
import streamlit as st
import joblib
import numpy as np

# 加载XGBoost模型
model = joblib.load('XGBoost-16.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "10975", "8787", "5491", "8463", "6838", "16044", "5374", "5619",
    "12361", "2996", "11642", "4940", "5680", "2844", "9864", "3857"
]

# Streamlit 用户界面
st.markdown("<h1 style='text-align: center; font-size: 24px;'>Predict Salmonella subtypes by integrating MALDI-TOF MS and XGBoost model</h1>", unsafe_allow_html=True)
st.write('Please enter the following indicators (in units of 10⁻⁶):')

# 创建两列布局
col1, col2 = st.columns(2)

# 在第一列输入前八个特征
with col1:
    input_10975 = st.number_input("10975±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=201, format="%d")
    input_8787 = st.number_input("8787±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=52, format="%d")
    input_5491 = st.number_input("5491±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=118, format="%d")
    input_8463 = st.number_input("8463±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=149, format="%d")
    input_6838 = st.number_input("6838±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=17, format="%d")
    input_16044 = st.number_input("16044±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=38, format="%d")
    input_5374 = st.number_input("5374±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=817, format="%d")
    input_5619 = st.number_input("5619±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=158, format="%d")

# 在第二列输入后八个特征
with col2:
    input_12361 = st.number_input("12361±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=33, format="%d")
    input_2996 = st.number_input("2996±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=127, format="%d")
    input_11642 = st.number_input("11642±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=14, format="%d")
    input_4940 = st.number_input("4940±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=5, format="%d")
    input_5680 = st.number_input("5680±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=41, format="%d")
    input_2844 = st.number_input("2844±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=33, format="%d")
    input_9864 = st.number_input("9864±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=0, format="%d")
    input_3857 = st.number_input("3857±1 (×10⁻⁶):", min_value=0, max_value=1000000, value=15, format="%d")

# 将输入的数据转化为实际模型需要的格式（除以10^6）
feature_values = [
    input_10975 / 1e6, input_8787 / 1e6, input_5491 / 1e6, input_8463 / 1e6,
    input_6838 / 1e6, input_16044 / 1e6, input_5374 / 1e6, input_5619 / 1e6,
    input_12361 / 1e6, input_2996 / 1e6, input_11642 / 1e6, input_4940 / 1e6,
    input_5680 / 1e6, input_2844 / 1e6, input_9864 / 1e6, input_3857 / 1e6
]

# 转换成模型的输入格式
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

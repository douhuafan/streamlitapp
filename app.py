import streamlit as st
import numpy as np
import joblib
import os

# 获取当前文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'extra_trees_classifier2.joblib')

# 加载预训练模型
clf_loaded = joblib.load(model_path)

# 定义特征
feature_cols = [ "Age",

    "Coronary hypoperfusion",
    "Cardiogenic shock",
    "D-dimer ",
    "fibrinogen degradathon product",
    "Lymphocytes",
    "Neutrophils",
    "creatinine",
    "glutamic oxaloacetic transaminase",
    "blood sugar",
    "muscle hemoglobin",
    "C-reaction protein",
    "hs-TNI",'LDH']

# Streamlit网页标题
st.title('Death Risk Prediction')

st.header('Input Features')

# 收集输入特征
input_features = []
for feature in feature_cols:
    # 修改number_input的format参数来避免自动四舍五入，并允许多于默认小数位数的输入
    value = st.number_input(feature, format="%.6f", step=0.000001)
    input_features.append(value)

# 将收集到的数据转换为适用于模型的格式
input_data = np.array(input_features).reshape(1, -1)
# 定义CSS样式，设置字体为Times New Roman并加粗
st.markdown("""
    <style>
    .custom-font {
        font-size:32px !important;  # 设置字号大小
        font-family: 'Times New Roman', Times, serif;  # 设置字体为Times New Roman
        font-weight: bold;  # 加粗字体
    }
    </style>
    """, unsafe_allow_html=True)
# 当用户点击“Predict”按钮时执行预测
if st.button('Predict'):
    prediction = clf_loaded.predict(input_data)
    prediction_proba = clf_loaded.predict_proba(input_data)

    # 显示预测结果
    st.subheader('Prediction Results:')
    predicted_class = prediction[0]
    # 若要显示特定类别（例如正类）的概率，可以选择该类别的概率
    predicted_proba_positive_class = prediction_proba[0][1]  # 假设正类为第二个类别
    percentage_proba=predicted_proba_positive_class* 100
    # 更改显示的概率值为特定类别的概率
    prob_text = f'Based feature values,predicted possibility of death is: {percentage_proba:.4f}%'
    st.markdown(f'<p class="custom-font">{prob_text}</p>', unsafe_allow_html=True)

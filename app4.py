import streamlit as st
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from filesplit.merge import Merge

path = os.getcwd().replace("\\", "/")

if os.path.exists(path+'/extra_trees_classifier4.joblib'):
    merge = Merge(path+"/model", path, 'extra_trees_classifier4.joblib')
else:
    pass

# 加载预训练模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'extra_trees_classifier4.joblib')

# 处理模型加载异常
try:
    clf_loaded = joblib.load(path+'/extra_trees_classifier4.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")

feature_cols = ['Age', 'Cardiogenic shock', 'Coronary hypoperfusion', 'hsTnI',
                'fibrinogen degradathon product', 'D-dimer ', 'blood sugar',
                'creatinine', 'fibrinogen', 'muscle hemoglobin',
                'standard bicarbonate concentration', 'Coronary artery disease']

st.title('Survival to Discharge Prediction')

st.header('Input Features')
st.set_option('deprecation.showPyplotGlobalUse', False)

input_features = []
for feature in feature_cols:
    value = st.number_input(feature, format="%.6f", step=0.000001)
    input_features.append(value)

input_data = np.array(input_features).reshape(1, -1)

st.markdown("""
    <style>
    .custom-font {
        font-size:32px !important;
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

if st.button('Predict'):
    prediction = clf_loaded.predict(input_data)
    prediction_proba = clf_loaded.predict_proba(input_data)

    st.subheader('Prediction Results:')
    predicted_class = prediction[0]
    predicted_proba_positive_class = prediction_proba[0][1]
    percentage_proba = predicted_proba_positive_class * 100
    prob_text = f'Based on feature values, predicted probability of Survival to discharge is: {percentage_proba:.4f}%'
    st.markdown(f'<p class="custom-font">{prob_text}</p>', unsafe_allow_html=True)

    explainer = shap.TreeExplainer(clf_loaded)
    shap_values = explainer.shap_values(input_data)

    # 绘制 SHAP Force Plot
    st.subheader('SHAP Force Plot:')
    shap.initjs()
    force_plot_html = shap.force_plot(
        explainer.expected_value[1],
        shap_values[0][:, 1],  # 选择正类 SHAP 值 (类别索引 1)
        input_data[0],
        feature_names=feature_cols
    )

    # 使用 HTML 组件展示
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
    components.html(force_html, height=300)

    # 绘制 SHAP Waterfall Plot
    # 创建 `shap.Explanation` 对象
    explanation = shap.Explanation(
        values=shap_values[0][:, 1],
        base_values=explainer.expected_value[1],
        data=input_data[0],
        feature_names=feature_cols
    )

    # 绘制 SHAP Waterfall Plot
    st.subheader('SHAP Waterfall Plot:')
    fig, ax = plt.subplots()
    shap.waterfall_plot(explanation)
    st.pyplot(fig)
    plt.clf()

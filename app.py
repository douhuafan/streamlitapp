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

st.markdown('''<style>
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="block-container"] {
        padding-bottom: 40px;
        padding-top: 40px;
    }
    .main {
        background-color: #f5f5f5d4;
    }
    [data-testid="stTextInput"] input {
        border: none !important;
        -webkit-box-shadow: inset 5px 5px 5px rgba(0, 0, 0, .2), inset -5px -5px 5px #fff;
        box-shadow: inset 5px 5px 5px rgba(0, 0, 0, .2), inset -5px -5px 5px #fff;
        //border-radius: 20px !important;
        //padding: 2em
    }
    [data-testid="stTextInput"],
    [data-testid="stExpander"] {
        border: none !important;
        -webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
        box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
        border-radius: 0.5rem !important;
        //text-align: center;
        padding: 1em
    }
    </style>''', unsafe_allow_html=True)
    
s = '''
border: none !important;
        -webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
        box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
        border-radius: 0.5rem;
        //text-align: center;
        padding: 1em;'''

# Streamlit网页标题
st.markdown(f'<h1 style="text-align: center; font-size: 25px; color: white; background: rgba(248,192,29); border-radius: .5rem; margin-bottom: 15px;{s}">Survival to discharge Prediction</h1>', unsafe_allow_html=True)
#st.title('Survival to discharge Prediction')

#st.header('Input Features')
expander = st.expander("**Input Features**", True)

# 收集输入特征
input_features = []
for feature in feature_cols:
    # 修改number_input的format参数来避免自动四舍五入，并允许多于默认小数位数的输入
    value = expander.number_input(feature, format="%.6f", step=0.000001)
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

col = expander.columns(5)
# 当用户点击“Predict”按钮时执行预测
if col[2].button('Predict', use_container_width=True):
    prediction = clf_loaded.predict(input_data)
    prediction_proba = clf_loaded.predict_proba(input_data)

    # 显示预测结果
    #st.subheader('Prediction Results:')
    with st.expander("**Prediction Results:**", True):
        predicted_class = prediction[0]
        # 若要显示特定类别（例如正类）的概率，可以选择该类别的概率
        predicted_proba_positive_class = prediction_proba[0][1]  # 假设正类为第二个类别
        percentage_proba=predicted_proba_positive_class* 100
        # 更改显示的概率值为特定类别的概率
        prob_text = f'Based feature values,predicted possibility of survival to discharge is: {percentage_proba:.4f}%'
        #st.markdown(f'<p class="custom-font">{prob_text}</p>', unsafe_allow_html=True)
        st.info("**"+prob_text+"**")
else:
    with st.expander("**Prediction Results:**", True):
        st.info("**Not start predict, you can click 'predict' button to start predict!**")

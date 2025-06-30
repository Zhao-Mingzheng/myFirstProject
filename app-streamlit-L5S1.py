import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="L5-S1 Prediction", layout="centered")

# 初始化 SHAP
shap.initjs()
plt.ioff()

# ✅ 加载模型和 scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("static/best_model_LR.joblib")
    # scaler = joblib.load("static/scaler.joblib")
    # scaler = joblib.load(r"D:\ST\web\web\static\standard_scaler.joblib")
    # return model, scaler
    return model

# ✅ 加载训练数据用于 TreeExplainer
@st.cache_data
def load_background_data():
    df = pd.read_excel("static/L5S1v11.xlsx", sheet_name='Sheet1').dropna()
    features = ['L5-S1 pfirrmann grade',
    'L5-S1 spinal canal stenosis',
    'L5-S1 modic change',
    'L5-S1 osteoarthritis of facet joints',
    'L5-S1 coronal imbalance',            # 新增这一行
    'L5-S1 foraminal stenosis',
    'L5-S1 EBQ',
    'L5-S1 local lordosis angle']
    df = df[features]
    return df, features

# ✅ SHAP 图转 base64
def plot_shap_image(shap_explainer, shap_values, input_data, base_value):
    # waterfall
    fig, _ = plt.subplots(figsize=(12, 5))
    exp = shap.Explanation(values=shap_values, base_values=base_value,
                           data=input_data.values[0], feature_names=input_data.columns)
    shap.plots.waterfall(exp, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=600)
    buf.seek(0)
    waterfall_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # force
    fig, _ = plt.subplots(figsize=(10, 4))
    # 这里不要 shap_values[0]，直接传 shap_values（一维向量）！
    shap.force_plot(base_value, shap_values, input_data, matplotlib=True, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=600)
    buf.seek(0)
    force_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return waterfall_b64, force_b64

# ✅ 主页面
def main():

    st.title("🔬 L5-S1 ASD Prediction")

    # model, scaler = load_model_scaler()
    model = load_model_scaler()
    background_df, feature_cols = load_background_data()

    st.sidebar.subheader('Input Parameters')
    # 用户输入
    user_input = {}
    for col in feature_cols:
        if ('spinal canal stenosis' in col
                or 'foraminal stenosis' in col
                or 'modic change' in col
                or 'osteoarthritis of facet joints' in col):
            user_input[col] = st.sidebar.selectbox(col, [0,1,2,3])
        elif 'pfirrmann grade' in col:
            user_input[col] = st.sidebar.selectbox(col, [1,2,3,4,5])
        elif 'sagittal imbalance' in col:
            user_input[col] = st.sidebar.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x else "No")
        elif 'coronal imbalance' in col:  # 新增
            user_input[col] = st.sidebar.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x else "No")
        elif 'preoperative disc height' in col:
            user_input[col] = st.sidebar.selectbox(col, [0, 1], format_func=lambda x: "≥10mm" if x else "<10mm")
        else:
            text_value = st.sidebar.text_input(col, value=5.0)
            user_input[col] = float(text_value)

    input_df = pd.DataFrame([user_input])
    if st.sidebar.button('Submit'):
        # 加载数据
        numerical_columns = ['L5-S1 EBQ', 'L5-S1 local lordosis angle']
        scaler = StandardScaler()
        background_df[numerical_columns] = scaler.fit_transform(background_df[numerical_columns])
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        print(input_df)
        # 预测
        # prediction = model.predict(input_df.to_numpy())
        proba = round(model.predict_proba(input_df.to_numpy())[0, 1], 4)
        print(proba)
        label = np.array([(proba >= 0.68).astype(int)])

        st.success(f"Base on feature values,predicted possibility of adjacent segment degeneration is {round(proba * 100, 2)}%.")

        # st.markdown(
        #     f"<h4 style='text-align: center;'>Base on feature values,predicted possibility of adjacent segment degeneration is {round(proba * 100, 2)}%.</h4>",
        #     unsafe_allow_html=True
        # )

        st.markdown(
            f"<h4 style='text-align: center;'>L5-S1 ASD Prediction Result = {label[0]}</h4>",
            unsafe_allow_html=True
        )

        # 解释模型
        # st.subheader("🧠 SHAP Explanation")
        # 先生成shap_values
        explainer = shap.KernelExplainer(model.predict_proba, background_df, link="logit")
        shap_values = explainer.shap_values(input_df, nsamples=100)

        if isinstance(shap_values, list):
            candidate = shap_values[1][0]
            print('candidate type:', type(candidate))
            print('candidate shape:', np.array(candidate).shape)
            # 强行变一维，只保留最后一维
            if len(np.array(candidate).shape) == 2 and np.array(candidate).shape[1] == 2:
                shap_vals = np.array(candidate)[:, 1]
                print('用[:, 1]后shap_vals.shape:', shap_vals.shape)
            elif len(np.array(candidate).shape) == 2:
                # 如果shape是(n,1)
                shap_vals = np.array(candidate).squeeze()
                print('squeeze后shap_vals.shape:', shap_vals.shape)
            else:
                shap_vals = np.array(candidate)
                print('直接用candidate, shap_vals.shape:', shap_vals.shape)
            bv = explainer.expected_value[1]
            base_value = float(bv[0]) if hasattr(bv, "__getitem__") else float(bv)
        else:
            # 明确只取类别1的SHAP值，把二维(8,2)变成一维(8,)
            shap_vals = np.array(shap_values[0])[:, 1]
            print('非list分支shap_vals.shape:', shap_vals.shape)  # 你会看到 (8,)
            bv = explainer.expected_value
            base_value = float(bv[0]) if hasattr(bv, "__getitem__") else float(bv)

        # 生成 SHAP 图
        w_b64, f_b64 = plot_shap_image(explainer, shap_vals, input_df, base_value)

        # 展示图像
        st.image(f"data:image/png;base64,{w_b64}", caption="Waterfall Plot")
        st.image(f"data:image/png;base64,{f_b64}", caption="Force Plot")

if __name__ == '__main__':
    main()
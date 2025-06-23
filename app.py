import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================
# Page Setup and Style
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="🔍"
)

st.markdown("""
    <style>
    .big-title {
        font-size: 45px !important;
        font-weight: bold;
        color: #003366;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 20px !important;
        color: #555;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #d3d3d3;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Customer Churn Prediction Using ANN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether a customer is likely to churn based on their profile using a trained ANN model.</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Load Model and Artifacts
# =========================
with st.spinner("🔄 Loading model and encoders..."):
    model = tf.keras.models.load_model(
        r'C:\Users\mudas\OneDrive\Desktop\Churn_modelling_ann_classification\notebooks\models\model.keras',
        compile=False
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with open(r'C:\Users\mudas\OneDrive\Desktop\Churn_modelling_ann_classification\notebooks\models\label_encoder.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open(r'C:\Users\mudas\OneDrive\Desktop\Churn_modelling_ann_classification\notebooks\models\ohe.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open(r'C:\Users\mudas\OneDrive\Desktop\Churn_modelling_ann_classification\notebooks\models\scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

# =========================
# Input Form - Left Column
# =========================
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📋 Customer Details")

    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    credit_score = st.number_input('📈 Credit Score', min_value=300, max_value=900, value=600)
    balance = st.number_input('💰 Balance', value=0.0)
    estimated_salary = st.number_input('💵 Estimated Salary', value=50000.0)
    tenure = st.slider('📆 Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('🛒 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('📊 Is Active Member', [0, 1])

# =========================
# Preprocess Input
# =========================
encoded_gender = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Gender': encoded_gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}])

input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_data)

# =========================
# Prediction
# =========================
prediction = model.predict(input_scaled)
churn_proba = prediction[0][0]

# =========================
# Output and Visualization
# =========================
with right_col:
    st.subheader("📊 Prediction Result")

    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.metric(label="🔍 Churn Probability", value=f"{churn_proba:.2%}")

        if churn_proba > 0.5:
            st.error("⚠️ The customer is **likely to churn.** Take action!")
        else:
            st.success("✅ The customer is **not likely to churn.**")

        

        # Dynamic labels and values
        labels = ['Churn', 'No Churn']
        values = [churn_proba, 1 - churn_proba]

        # Dynamic colors: red if churn is high, green otherwise
        colors = ['#FF4B4B', '#00B050'] if churn_proba > 0.5 else ['#00B050', '#FF4B4B']

        # Create the figure
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#000000', width=2)),
            hoverinfo='label+percent',
            textinfo='label+percent',
            textfont_size=18
        )])

        # Update layout for cleaner look
        fig.update_layout(
            title_text='📈 Churn vs No Churn Distribution',
            title_font_size=20,
            showlegend=True,
            margin=dict(t=40, b=20, l=10, r=10)
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        st.markdown('</div>', unsafe_allow_html=True)

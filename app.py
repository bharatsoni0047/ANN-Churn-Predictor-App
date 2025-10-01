import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import warnings

# turn off warnings
warnings.filterwarnings("ignore")

# load model, encoders, and scaler only once
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model("model.h5")
    with open("onehot_encoder_geo.pkl", "rb") as f:
        ohe = pickle.load(f)  # one-hot encoder for geography
    with open("label_encoder_gender.pkl", "rb") as f:
        le = pickle.load(f)   # label encoder for gender
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # scaler for numeric values
    return model, ohe, le, scaler

model, ohe, le, scaler = load_model_and_encoders()

# app title
st.title("💳 Customer Churn Prediction App")
st.write("Fill in the details below to see churn chance:")

# user input
geography = st.selectbox("🌍 Geography", ohe.categories_[0])
gender = st.selectbox("👤 Gender", le.classes_)
age = st.slider("🎂 Age", 18, 90, 30)
balance = st.number_input("💰 Balance", 0.0, 1e7, 0.0)
credit_score = st.number_input("💳 Credit Score", 0, 1000, 600)
salary = st.number_input("💵 Estimated Salary", 0.0, 1e6, 50000.0)
tenure = st.slider("📅 Tenure (Years with Bank)", 0, 10, 5)
num_products = st.slider("🛒 Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member", [0, 1])

# prepare input for the model
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le.transform([gender])[0]],  # convert gender to number
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [salary]
})

# convert geography to one-hot encoding
geo_ohe = ohe.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_ohe, columns=ohe.get_feature_names_out(["Geography"]))

# combine numeric + one-hot features
input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

# scale input like training
input_scaled = scaler.transform(input_df)

# try-except for safe prediction
try:
    pred = model.predict(input_scaled)[0][0]  # make prediction
    st.subheader(f"📊 Churn Probability: **{pred:.2f}**")
    if pred > 0.5:
        st.error("⚠️ This customer has high chance of churn")
    else:
        st.success("✅ This customer has low chance of churn")
except Exception as e:
    st.error(f"Something went wrong during prediction: {e}")

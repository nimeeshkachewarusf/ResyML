import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("craving_model_dish1.pkl")

# Title
st.title("Craving Prediction for December (Dish 1)")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV of user transaction history")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Assume monthly binary strings exist from Jan to Nov
    for month in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']:
        df[month] = df[month].apply(lambda x: str(int(x)).zfill(15))
        for i in range(15):
            df[f"{month}_dish{i+1}"] = df[month].str[i].astype(int)

    # Select only the required feature columns
    features = [f"{month}_dish{i+1}" for month in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov'] for i in range(15)]
    X = df[features]

    # Predict
    preds = model.predict(X)
    df["Predicted_Craving_Dish1"] = preds

    # Show results
    st.write("Prediction Results:")
    st.dataframe(df[["user_id", "Predicted_Craving_Dish1"]])

    # Optional: Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction CSV", csv, "craving_predictions.csv", "text/csv")
import streamlit as st
import pandas as pd
import joblib
import os

# --- Load all 15 models ---
models = {}
for i in range(1, 16):
    model_path = f"craving_model_dish{i}.pkl"
    if os.path.exists(model_path):
        models[f'dish{i}'] = joblib.load(model_path)

# --- App Title ---
st.title("Craving Prediction for 15 Dishes 🍽️")
st.markdown("Upload a user profile CSV to get predictions or choose a user ID below.")

# --- Upload CSV or select user ---
uploaded_file = st.file_uploader("Upload user profile CSV", type=["csv"])
user_data = None

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    user_ids = user_df['user_id'].tolist()
    selected_user = st.selectbox("Select a user ID", user_ids)
    user_data = user_df[user_df['user_id'] == selected_user].drop(columns=['user_id'])
else:
    st.warning("Please upload a CSV file containing user profile(s).")

# --- Prediction ---
if user_data is not None and st.button("Predict Cravings"):
    results = []
    for dish, model in models.items():
        prob = model.predict_proba(user_data)[0][1]  # Probability of craving (class 1)
        prediction = 1 if prob >= 0.5 else 0
        results.append({
            "Dish": dish,
            "Craving (Yes=1)": prediction,
            "Probability": f"{prob*100:.2f} %"
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # --- Download CSV ---
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="craving_predictions.csv",
        mime='text/csv'
    )
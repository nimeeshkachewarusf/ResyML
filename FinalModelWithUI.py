import gradio as gr
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# -- Load Data (place these at top-level for efficiency)
PARQUET_FILE = "output_with_userids_usernames.parquet"
MODEL_FILE = "lgbm_dish_recommender_with_history.pkl"
USER_LOCATIONS_FILE = "Users.csv"
RESTAURANT_FILE = "restaurants_with_cuisine.csv"
N_PREV_MONTHS = 6
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15

dish_to_cuisine = {
    'dish1': 'Indian', 'dish2': 'Indian', 'dish3': 'Indian', 'dish4': 'Indian', 'dish5': 'Indian',
    'dish6': 'Japanese', 'dish7': 'Japanese', 'dish8': 'Japanese', 'dish9': 'Japanese', 'dish10': 'Japanese',
    'dish11': 'Chinese', 'dish12': 'Chinese', 'dish13': 'Chinese', 'dish14': 'Chinese', 'dish15': 'Chinese'
}
dish_to_name = {
    'dish1': "Aamras Poori", 'dish2': "Boondi Raita", 'dish3': "Pakora", 'dish4': "Sarson da Saag & Makki di Roti", 'dish5': "Pav Bhaji",
    'dish6': "Sushi Rolls", 'dish7': "Matcha Ice Cream", 'dish8': "Udon Noodles", 'dish9': "Tonkotsu Ramen", 'dish10': "Gyoza",
    'dish11': "Chilli Paneer", 'dish12': "Gobi Manchurian", 'dish13': "Chilli Mushroom", 'dish14': "American Chopsuey", 'dish15': "Schezwan Fried Rice"
}     # fill in as before

df = pd.read_parquet(PARQUET_FILE)
model_package = joblib.load(MODEL_FILE)
user_locs = pd.read_csv(USER_LOCATIONS_FILE)
restaurants = pd.read_csv(RESTAURANT_FILE)

def bitstring_to_array(s):
    return np.array([int(ch) for ch in str(s).zfill(N_DISHES)])

def build_user_features(user_row, month_to_predict):
    t = MONTHS.index(month_to_predict)
    if t < N_PREV_MONTHS:
        raise ValueError(f"Not enough history")
    feats = []
    for w in range(N_PREV_MONTHS):
        m = MONTHS[t - w - 1]
        feats.extend(bitstring_to_array(user_row[m]))
        feats.extend(bitstring_to_array(user_row[f"{m}_craving"]))
    month_onehot = [0]*len(MONTHS)
    month_onehot[t] = 1
    feats.extend(month_onehot)
    return np.array(feats).reshape(1, -1)

def recommend(user_id, month_to_predict, top_k):
    # Get user info & location
    user_rows = df[df['UserID'].astype(str) == str(user_id)]
    if user_rows.empty:
        return "UserID not found", []
    user_row = user_rows.iloc[0]
    user_loc_row = user_locs[user_locs['UserID'].astype(str) == str(user_id)]
    if user_loc_row.empty:
        return "User location not found", []
    user_lat = float(user_loc_row.iloc[0]['Latitude'])
    user_lon = float(user_loc_row.iloc[0]['Longitude'])
    # Recommend dishes
    try:
        features = build_user_features(user_row, month_to_predict)
    except:
        return "Not enough history for selected month", []
    model = model_package['model']
    thresholds = model_package['thresholds']
    dish_names = [f'dish{i+1}' for i in range(N_DISHES)]
    proba = np.array([p[:,1] for p in model.predict_proba(features)]).T[0]
    predicted = (proba > thresholds).astype(int)
    if top_k:
        top_indices = np.argsort(proba)[-top_k:][::-1]
        recommended = [dish_names[i] for i in top_indices]
    else:
        recommended = [dish for dish, p in zip(dish_names, predicted) if p]
    # Prepare results table
    recs_out = []
    for dish in recommended:
        recs_out.append(f"{dish_to_name[dish]} (Cuisine: {dish_to_cuisine[dish]})")
    # Prepare nearby restaurants for each cuisine
    output_tables = []
    cuisines = set(dish_to_cuisine[dish] for dish in recommended)
    for cuisine in cuisines:
        subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
        subset['DistanceKM'] = subset.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km, axis=1
        )
        top_nearby = subset.sort_values('DistanceKM').head(5)
        rest_entries = [
            f"{row['RestaurantName']} — {row['DistanceKM']:.2f} km"
            for _, row in top_nearby.iterrows()
        ]
        output_tables.append((cuisine, rest_entries))
    # Join for display
    rest_out = ""
    for cuisine, rest_list in output_tables:
        rest_out += f"---\nNearest {cuisine} restaurants:\n" + "\n".join(rest_list) + "\n"
    return "\n".join(recs_out), rest_out

months_dropdown = gr.Dropdown(choices=MONTHS, label="Month to Predict")
inputs = [
    gr.Textbox(label="UserID"),
    months_dropdown,
    gr.Number(label="Top K Dishes to Recommend (optional)", value=None, precision=0)
]

outputs = [
    gr.Textbox(label="Recommended Dishes"),
    gr.Textbox(label="Nearby Restaurants"),
]

demo = gr.Interface(
    fn=recommend,
    inputs=inputs,
    outputs=outputs,
    title="Personalized Restaurant & Dish Recommender",
    description="Enter UserID and month to get recommendations based on your eating and craving history, plus the nearest restaurants by cuisine."
)
demo.launch(share=True)  # Set share=True to get a public link!

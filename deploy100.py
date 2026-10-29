import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# ----------- CONFIGURATION -----------
PARQUET_FILE = "output_with_userids_usernames.parquet"
MODEL_FILE = "boosted_dish_recommender.pkl"
RESTAURANTS_FILE = "restaurants.csv"  # must have RestaurantName, Cuisine, Latitude, Longitude
N_PREV_MONTHS = 6
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15
TOP_N_RECOMMEND = 5

# Dish metadata for mapping
dish_to_cuisine = {
    'dish1': 'Indian', 'dish2': 'Indian', 'dish3': 'Indian', 'dish4': 'Indian', 'dish5': 'Indian',
    'dish6': 'Japanese', 'dish7': 'Japanese', 'dish8': 'Japanese', 'dish9': 'Japanese', 'dish10': 'Japanese',
    'dish11': 'Chinese', 'dish12': 'Chinese', 'dish13': 'Chinese', 'dish14': 'Chinese', 'dish15': 'Chinese'
}
dish_to_name = {
    'dish1': "Aamras Poori", 'dish2': "Boondi Raita", 'dish3': "Pakora", 'dish4': "Sarson da Saag & Makki di Roti", 'dish5': "Pav Bhaji",
    'dish6': "Sushi Rolls", 'dish7': "Matcha Ice Cream", 'dish8': "Udon Noodles", 'dish9': "Tonkotsu Ramen", 'dish10': "Gyoza",
    'dish11': "Chilli Paneer", 'dish12': "Gobi Manchurian", 'dish13': "Chilli Mushroom", 'dish14': "American Chopsuey", 'dish15': "Schezwan Fried Rice"
}

def bitstring_to_array(s):
    return np.array([int(ch) for ch in str(s).zfill(N_DISHES)])

def get_user_features(user_row, month_to_predict):
    t = MONTHS.index(month_to_predict)
    feats = []
    for w in range(N_PREV_MONTHS):
        m_idx = t - w - 1
        if m_idx >= 0:
            m = MONTHS[m_idx]
            feats.extend(bitstring_to_array(user_row[m]))
            feats.extend(bitstring_to_array(user_row[f"{m}_craving"]))
        else:
            feats.extend([0]*N_DISHES)
            feats.extend([0]*N_DISHES)
    # Add one-hot for prediction month
    month_onehot = [0]*len(MONTHS)
    month_onehot[t] = 1
    feats.extend(month_onehot)
    return np.array(feats).reshape(1, -1)

def recommend_dishes(user_row, month_to_predict, model_package, top_k=None):
    features = get_user_features(user_row, month_to_predict)
    model = model_package['model']
    thresholds = model_package['thresholds']
    dish_names = [f'dish{i+1}' for i in range(N_DISHES)]
    # Model predicts per-dish probability
    proba = np.array([p[:,1] for p in model.predict_proba(features)]).T[0]
    # Optionally: top_k or by threshold
    if top_k is not None:
        top_indices = np.argsort(proba)[-top_k:][::-1]
        recommended = [dish_names[i] for i in top_indices]
    else:
        predicted = (proba > thresholds).astype(int)
        recommended = [dish for dish, p in zip(dish_names, predicted) if p]
    # Remove duplicates while preserving order
    recommended = list(dict.fromkeys(recommended))
    return recommended, proba

def find_nearby_cuisine_restaurants(user_lat, user_lon, cuisines, N=5):
    restaurants = pd.read_csv(RESTAURANTS_FILE)
    restaurants.columns = restaurants.columns.str.strip()  # Remove spaces in header
    found = {}
    for cuisine in cuisines:
        subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
        subset['DistanceKM'] = subset.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km, axis=1
        )
        top_nearby = subset.sort_values('DistanceKM').head(N)
        found[cuisine] = top_nearby
    return found

# ---- LOAD MODEL AND DATA ----
df = pd.read_parquet(PARQUET_FILE)
model_package = joblib.load(MODEL_FILE)

# ---- USER INPUT ----
userid = input(f"Enter UserID: ").strip()
month_to_predict = input(f"Enter month to predict (e.g. 'Aug'): ").strip()
top_k = input("How many top dishes to recommend? (press Enter for all passing threshold): ").strip()
top_k = int(top_k) if top_k.isnumeric() else TOP_N_RECOMMEND

# For demo: ask user for their lat/lon
user_lat = float(input("Enter your latitude: ").strip())
user_lon = float(input("Enter your longitude: ").strip())

user_rows = df[df['UserID'].astype(str) == str(userid)]
if user_rows.empty:
    raise ValueError(f"UserID '{userid}' not found in the data!")
user_row = user_rows.iloc[0]

# ---- RECOMMEND DISHES ----
recommended, proba = recommend_dishes(user_row, month_to_predict, model_package, top_k=top_k)
print(f"\nRecommended dishes for UserID {userid} in {month_to_predict}:")
for dish in recommended:
    print(f"  {dish_to_name[dish]} (Cuisine: {dish_to_cuisine[dish]})")

# ---- RESTAURANT SUGGESTIONS BY CUISINE ----
cuisines_for_this_user = set(dish_to_cuisine[d] for d in recommended)
restaurant_lists = find_nearby_cuisine_restaurants(user_lat, user_lon, cuisines_for_this_user, N=5)
for cuisine, near_df in restaurant_lists.items():
    print(f"\nRestaurants serving {cuisine} cuisine near you:")
    for _, row in near_df.iterrows():
        print(f"  {row['RestaurantName']} — {row['DistanceKM']:.2f} km away (Lat: {row['Latitude']}, Lon: {row['Longitude']})")

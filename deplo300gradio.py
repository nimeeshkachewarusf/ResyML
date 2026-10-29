import pandas as pd
import numpy as np
import joblib

# ----- CONFIG (same as training) -----
PARQUET_FILE = "output_with_userids_usernames.parquet"
MODEL_FILE = "lgbm_dish_recommender_with_history.pkl"
N_PREV_MONTHS = 6
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15

def bitstring_to_array(s):
    return np.array([int(ch) for ch in str(s).zfill(N_DISHES)])

def build_user_features(user_row, month_to_predict):
    t = MONTHS.index(month_to_predict)
    if t < N_PREV_MONTHS:
        raise ValueError(f"Not enough history to predict {month_to_predict} (need at least {N_PREV_MONTHS} prior months)")
    feats = []
    # Use past N_PREV_MONTHS months (most recent first)
    for w in range(N_PREV_MONTHS):
        m = MONTHS[t - w - 1]
        feats.extend(bitstring_to_array(user_row[m]))
        feats.extend(bitstring_to_array(user_row[f"{m}_craving"]))
    # Add one-hot for the prediction month
    month_onehot = [0]*len(MONTHS)
    month_onehot[t] = 1
    feats.extend(month_onehot)
    return np.array(feats).reshape(1, -1)

def recommend_dishes(user_row, month_to_predict, model_package, top_k=None):
    features = build_user_features(user_row, month_to_predict)
    model = model_package['model']
    thresholds = model_package['thresholds']
    dish_names = [f'dish{i+1}' for i in range(N_DISHES)]
    # Get predicted probabilities for each dish
    proba = np.array([p[:,1] for p in model.predict_proba(features)]).T[0]
    # Thresholding
    predicted = (proba > thresholds).astype(int)
    # Optionally, recommend only top_k dishes
    if top_k is not None:
        # Overwrite with top-k highest probabilities
        top_indices = np.argsort(proba)[-top_k:][::-1]
        recommended = [dish_names[i] for i in top_indices]
    else:
        recommended = [dish for dish, p in zip(dish_names, predicted) if p]
    # Also show probabilities if desired
    for i, dish in enumerate(dish_names):
        print(f"{dish:12s}  proba={proba[i]:.3f}  threshold={thresholds[i]:.2f}  predicted={'YES' if predicted[i] else 'NO'}")
    return recommended

# ----- LOAD DATA -----
df = pd.read_parquet(PARQUET_FILE)
model_package = joblib.load(MODEL_FILE)

# ----- USER INPUT -----
userid = input(f"Enter UserID: ").strip()
month_to_predict = input(f"Enter month to predict (e.g. 'Aug'): ").strip()
top_k = input("How many top dishes to recommend? (press Enter for all passing threshold): ").strip()
top_k = int(top_k) if top_k.isnumeric() else None

# ----- GET USER & PREDICT -----
user_rows = df[df['UserID'].astype(str) == str(userid)]
if user_rows.empty:
    raise ValueError(f"UserID '{userid}' not found in the data!")
user_row = user_rows.iloc[0]

recommended = recommend_dishes(user_row, month_to_predict, model_package, top_k=top_k)
print(f"\nRecommended dishes for UserID {userid} in {month_to_predict}:")
for dish in recommended:
    print("  ", dish)


######################################################################################################
from geopy.distance import geodesic
import pandas as pd

# Maps 'dishX' to cuisine (as per your previous setup)
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


user_lat, user_lon = 19.076, 72.8777  # Replace with actual user location!
restaurants = pd.read_csv('restaurants_with_cuisine.csv')  # Must have Cuisine, Latitude, Longitude, RestaurantName

# Get cuisines from recommendations
cuisines_for_this_user = set(dish_to_cuisine[dish] for dish in recommended)

print("\n--- Dish & Cuisine Recommendations ---")
for dish in recommended:
    print(f"{dish_to_name[dish]} (Cuisine: {dish_to_cuisine[dish]})")

for cuisine in cuisines_for_this_user:
    subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
    subset['DistanceKM'] = subset.apply(
        lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km, axis=1
    )
    top_nearby = subset.sort_values('DistanceKM').head(5)
    print(f"\nRestaurants serving {cuisine} cuisine near you:")
    for i, row in top_nearby.iterrows():
        print(f"  {row['RestaurantName']} — {row['DistanceKM']:.2f} km away")



import gradio as gr

def recommend_ui(userid, month, top_k):
    try:
        userid_str = str(userid).strip()
        month_str = str(month).strip()
        k_val = int(top_k) if top_k is not None and str(top_k).strip() != "" and float(top_k) == int(float(top_k)) else None
    except Exception:
        return "Invalid input!"

    # Find user
    matches = df[df['UserID'].astype(str) == userid_str]
    if matches.empty:
        return f"UserID '{userid_str}' not found in the data!"
    user_row = matches.iloc[0]

    try:
        # Get dish recommendations
        recommended_dishes = recommend_dishes(user_row, month_str, model_package, top_k=k_val)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return "Prediction failed: " + str(e)

    if not recommended_dishes:
        return "No dishes recommended."
    out = f"Recommended dishes for UserID {userid_str} in {month_str}:\n"
    cuisines_for_this_user = set()
    for dish in recommended_dishes:
        dish_name = dish_to_name.get(dish, dish)
        cuisine = dish_to_cuisine.get(dish, 'Unknown')
        cuisines_for_this_user.add(cuisine)
        out += f"  {dish_name} (Cuisine: {cuisine})\n"

    # Use fixed location (or accept as input if you wish)
    user_lat, user_lon = 19.076, 72.8777

    for cuisine in cuisines_for_this_user:
        try:
            subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
            subset['DistanceKM'] = subset.apply(
                lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km, axis=1
            )
            top_nearby = subset.sort_values('DistanceKM').head(5)
            if not top_nearby.empty:
                out += f"\nRestaurants serving {cuisine} cuisine near you:\n"
                for i, row in top_nearby.iterrows():
                    out += f"  {row['RestaurantName']} — {row['DistanceKM']:.2f} km away\n"
            else:
                out += f"\nNo restaurants found serving {cuisine} cuisine near you.\n"
        except Exception as e:
            out += f"\nError retrieving restaurants for {cuisine} cuisine: {str(e)}\n"

    return out



interface = gr.Interface(
    fn=recommend_ui,
    inputs=[
        gr.Text(label="UserID"),
        gr.Text(label="Month (e.g. 'Aug')"),
        gr.Number(label="Top K Dishes", value=None)
    ],
    outputs="text",
    title="Restaurant Recommender"
)

interface.launch()
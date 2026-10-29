import numpy as np
import pandas as pd
import random

# --- Parameters ---
n_users = 100_000
n_restaurants_per_user = 1
dishes = [
    "Aamras Poori", "Boondi Raita", "Pakora", "Sarson da Saag & Makki di Roti", "Pav Bhaji",
    "Sushi Rolls", "Matcha Ice Cream", "Udon Noodles", "Tonkotsu Ramen", "Gyoza",
    "Chilli Paneer", "Gobi Manchurian", "Chilli Mushroom", "American Chopsuey", "Schezwan Fried Rice"
]
cuisines = ["Indian", "Japanese", "Chinese"]
cuisine_dish_idx = {
    "Indian": list(range(0,5)),
    "Japanese": list(range(5,10)),
    "Chinese": list(range(10,15)),
}
season_months = {
    "Summer": [4,5,6],      # May, Jun, Jul
    "Monsoon": [7,8,9],     # Aug, Sep, Oct
    "Winter": [0,1,2],      # Jan, Feb, Mar
}
season_hot_dish_idx = {
    "Summer": [0,1,5,6,10],
    "Winter": [2,7,8,13,14],
    "Monsoon": [3,4,9,11,12],
}
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def get_season(month_idx):
    for season, mlist in season_months.items():
        if month_idx in mlist:
            return season
    return "Other"

rows = []
for user in range(n_users):
    cuisine = random.choice(cuisines)
    dishes_idx = cuisine_dish_idx[cuisine]
    row = {}
    for m_idx, mon in enumerate(months):
        season = get_season(m_idx)
        dish_probs = np.full(15, 0.10)
        for idx in dishes_idx:
            dish_probs[idx] = 0.25
        for idx in season_hot_dish_idx.get(season, []):
            if idx in dishes_idx:
                dish_probs[idx] = 0.65
        for idx in range(15):
            if idx not in dishes_idx:
                dish_probs[idx] = 0.0
        eat_this_month = (np.random.rand(15) < dish_probs).astype(int)
        crave_this_month = np.zeros(15, dtype=int)
        for i_dish in range(15):
            if eat_this_month[i_dish] and random.random() < 0.7:
                crave_this_month[i_dish] = 1
        row[mon] = ''.join(str(x) for x in eat_this_month)
        row[f"{mon}_craving"] = ''.join(str(x) for x in crave_this_month)
    rows.append(row)
    if user % 10000 == 0 and user > 0:
        print(f"Generated {user} rows.")

# Assemble DataFrame
df = pd.DataFrame(rows)
print("Sample rows:\n", df.head())

# --- Save as CSV and Parquet ---
df.to_csv("synthetic_restaurant_flat_100k.csv", index=False)
df.to_parquet("synthetic_restaurant_flat_100k.parquet", index=False)
print("Saved to:")
print(" - CSV:      synthetic_restaurant_flat_100k.csv")
print(" - Parquet:  synthetic_restaurant_flat_100k.parquet")
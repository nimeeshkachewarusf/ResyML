import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from geopy.distance import geodesic
import gradio as gr

# --- CONFIGURATION ---
PARQUET_FILE = "flattened_by_seaons_with_userinfo.parquet"
RESTAURANT_FILE = "restaurants_with_cuisine.csv"
USER_LOC_FILE = "Users.csv"
N_DISHES = 15
LATENT_DIM = 8
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

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

# --- LOAD DATA ---
df = pd.read_parquet(PARQUET_FILE)
restaurants = pd.read_csv(RESTAURANT_FILE)
user_locs = pd.read_csv(USER_LOC_FILE)

# --- MODEL DEFINITION (must match training) ---
class DishAutoencoder(nn.Module):
    def __init__(self, n_dishes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_dishes, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_dishes),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# --- LOAD TRAINED AUTOENCODER ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DishAutoencoder(N_DISHES, LATENT_DIM).to(device)
model.load_state_dict(torch.load("autoencoder_recommender.pt", map_location=device))
model.eval()

# --- PREP USER INPUT --
def get_user_input_vec(user_id, month):
    # Find row for user_id
    user_rows = df[df['UserID'].astype(str) == str(user_id)]
    if user_rows.empty:
        return None
    user_row = user_rows.iloc[0]
    # Use the prior history for the season/month as in your pipeline
    # Example: just use the dish vector for that month (or combine with context features if you wish)
    month_vector = [int(ch) for ch in str(user_row[month]).zfill(N_DISHES)]
    return np.array(month_vector, dtype=np.float32)

def get_user_location(user_id):
    row = user_locs[user_locs['UserID'].astype(str) == str(user_id)]
    if row.empty:
        return None, None
    return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])

# --- RECOMMENDATION FUNCTION ---
def recommend(user_id, month, top_k=5, threshold=0.5):
    # Input vector for user + month
    user_vec = get_user_input_vec(user_id, month)
    if user_vec is None:
        return "UserID or month not found.", ""
    lat, lon = get_user_location(user_id)
    if lat is None or lon is None:
        return "User location not found.", ""
    # Autoencoder inference
    inp = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp).cpu().numpy().flatten()
    # Pick top-K or threshold
    top_indices = np.argsort(out)[-top_k:][::-1]
    recommended = [(f'dish{i+1}', out[i]) for i in top_indices]
    # Human-readable names/cuisines
    rec_lines = []
    cuisines = set()
    for dish, score in recommended:
        name = dish_to_name[dish]
        cuisine = dish_to_cuisine[dish]
        rec_lines.append(f"{name} (Cuisine: {cuisine}) — score: {score:.2f}")
        cuisines.add(cuisine)
    # Restaurant recommendations by cuisine
    rest_out = ""
    for cuisine in cuisines:
        subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
        subset['DistanceKM'] = subset.apply(
            lambda row: geodesic((lat, lon), (row['Latitude'], row['Longitude'])).km, axis=1
        )
        top_nearby = subset.sort_values('DistanceKM').head(5)
        rest_out += f"\n---\nNearest {cuisine} restaurants:\n"
        for _, row in top_nearby.iterrows():
            rest_out += f"{row['RestaurantName']} — {row['DistanceKM']:.2f} km\n"
    return "\n".join(rec_lines), rest_out.strip()

# --- GRADIO DEPLOYMENT ---
iface = gr.Interface(
    fn=lambda userid, month: recommend(userid, month, top_k=5),
    inputs=[
        gr.Textbox(label='UserID'),
        gr.Dropdown(choices=MONTHS, label="Month to Predict")
    ],
    outputs=[
        gr.Textbox(label="Recommended Dishes"),
        gr.Textbox(label="Nearby Restaurants"),
    ],
    title="Autoencoder Restaurant & Dish Recommender",
    description="Enter UserID and month to get dish and restaurant recommendations (autoencoder powered)."
)

iface.launch()

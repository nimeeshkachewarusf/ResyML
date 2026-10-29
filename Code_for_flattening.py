import pandas as pd
import random

# Load the source data
df = pd.read_csv("test1.csv", dtype=str)

# Define dishes and seasons
cuisine_dishes = {
    "Indian": ["Aamras Poori", "Boondi Raita", "Pakora", "Sarson da Saag & Makki di Roti", "Pav Bhaji"],
    "Japanese": ["Sushi Rolls", "Matcha Ice Cream", "Udon Noodles", "Tonkotsu Ramen", "Gyoza"],
    "Chinese": ["Chilli Paneer", "Gobi Manchurian", "Chilli Mushroom", "American Chopsuey", "Schezwan Fried Rice"]
}
all_dishes = cuisine_dishes["Indian"] + cuisine_dishes["Chinese"] + cuisine_dishes["Japanese"]

seasonal_map = {
    "Jan": "Winter", "Feb": "Winter", "Mar": "Summer", "Apr": "Summer",
    "May": "Summer", "Jun": "Monsoon", "Jul": "Monsoon", "Aug": "Monsoon",
    "Sep": "Monsoon", "Oct": "Winter", "Nov": "Winter", "Dec": "Winter"
}
seasonal_dishes = {
    "Summer": ["Aamras Poori", "Boondi Raita", "Sushi Rolls", "Matcha Ice Cream", "Chilli Paneer"],
    "Winter": ["Pakora", "Udon Noodles", "Tonkotsu Ramen", "American Chopsuey", "Schezwan Fried Rice"],
    "Monsoon": ["Sarson da Saag & Makki di Roti", "Pav Bhaji", "Gyoza", "Gobi Manchurian", "Chilli Mushroom"]
}
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Flexible 15-digit string generator
def generate_anydish_string(month):
    season = seasonal_map[month]
    hot_dishes = seasonal_dishes[season]
    return "".join(
        "1" if random.random() < (0.7 if dish in hot_dishes else 0.3) else "0"
        for dish in all_dishes
    )

# Craving generator (70-80% match)
def generate_craving_string(trans_str, min_overlap=0.7, max_overlap=0.8):
    length = len(trans_str)
    overlap_count = random.randint(int(min_overlap * length), int(max_overlap * length))
    ones_indices = [i for i, ch in enumerate(trans_str) if ch == '1']
    keep_ones = set(random.sample(ones_indices, min(len(ones_indices), overlap_count)))
    return ''.join('1' if i in keep_ones else '0' for i in range(length))

# Create dataset with 200,000 rows
df_clean = df.dropna(subset=["resname"])
df_sampled = df_clean.sample(n=100000, replace=True).reset_index(drop=True)
df_sampled = pd.concat([df_sampled, df_sampled.copy()], ignore_index=True)

# Generate final dataset
final_rows = []
for _, row in df_sampled.iterrows():
    record = {
        "user_id": row["user_id"],
        "username": row["username"],
        "restaurant_id": row["restaurant_id"],
        "resname": row["resname"]
    }
    for month in months:
        trans = generate_anydish_string(month)
        record[month] = trans
        record[f"{month}_craving"] = generate_craving_string(trans)
    final_rows.append(record)

# Save to file
final_df = pd.DataFrame(final_rows)
file_path = "final_flattened_transactions_flexible.csv"
final_df.to_csv(file_path, index=False)

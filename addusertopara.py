import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Load your existing Parquet file with transaction history
df_transactions = pd.read_parquet('synthetic_restaurant_flat_100k.parquet')

# Number of users (assumed equal to number of rows)
num_users = 100000

# Generate user IDs: 1 to num_users
user_ids = list(range(1, num_users + 1))

# Generate usernames using Faker
# To ensure usernames roughly correspond 1:1 to user ids, generate num_users unique usernames
# Faker generates random names, here we use user_name() method for realistic usernames
usernames = [fake.user_name() for _ in range(num_users)]

# Add columns to your dataframe
df_transactions['UserID'] = user_ids
df_transactions['UserName'] = usernames

# Save the updated DataFrame back to Parquet
df_transactions.to_parquet('transactions_with_userids&names.parquet', index=False)

print("UserIDs and Usernames added and saved successfully!")
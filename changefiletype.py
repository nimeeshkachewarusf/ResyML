import pandas as pd
from faker import Faker

# --------- 1. Load ----------
filename = 'synthetic_restaurant_flat_100k.parquet'  # or .csv

if filename.endswith('.parquet'):
    df = pd.read_parquet(filename)
else:
    df = pd.read_csv(filename, dtype=str)

# --------- 2. Add user columns ----------
num_users = len(df)
df['UserID'] = range(1, num_users + 1)
fake = Faker()
df['UserName'] = [fake.name() for _ in range(num_users)]

# --------- 3. Save ----------
df.to_parquet('output_with_userids_usernames.parquet', index=False)
df.to_csv('output_with_userids_usernames.csv', index=False)

# --------- 4. Verify ----------
print(df[['UserID', 'UserName']].head())

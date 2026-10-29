import pandas as pd

df_parquet = pd.read_parquet("flattened_by_seaons.parquet")
df_users = pd.read_csv("users.csv")  # Must have UserID and UserName!

df_parquet["UserID"] = df_users["UserID"]
df_parquet["UserName"] = df_users["UserName"]

df_parquet.to_parquet("flattened_by_seaons_with_userinfo.parquet", index=False)

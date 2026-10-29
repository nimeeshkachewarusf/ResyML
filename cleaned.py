import pandas as pd

def clean_bitstring(s):
   
    s = str(s)
    digits = ''.join(ch for ch in s if ch in '01')
    if len(digits) >= 15:
        return digits[:15]
    return digits.rjust(15, '0')

input_file = "final_filtered_with_cravings.csv"
output_file = "filtered_with_cravings_cleaned.csv"

# Load csv, forcing all columns to str to preserve all info
df = pd.read_csv(input_file, dtype=str)

# Optionally, define which columns to clean
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
cols_to_clean = [col for col in df.columns if any(month in col for month in months)]

# Clean all relevant columns in place
for col in cols_to_clean:
    df[col] = df[col].apply(clean_bitstring)

# Save the cleaned dataframe to a new file
df.to_csv(output_file, index=False)

print(f"Cleaned file saved as {output_file}")
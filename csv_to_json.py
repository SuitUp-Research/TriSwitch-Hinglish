import pandas as pd
import json

# Read the CSV file
csv_file = "llama_translations.csv"
output_file = "llama_translations.json"

print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file)

# Convert to JSON
print(f"Converting to JSON...")
data = df.to_dict(orient='records')

# Save to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} records to {output_file}")

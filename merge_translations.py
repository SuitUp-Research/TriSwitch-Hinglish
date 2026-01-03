import pandas as pd
import json

# Load base dataset
print("Loading base dataset...")
with open('dataset/db_with_reference_en.json', 'r', encoding='utf-8') as f:
    base_data = json.load(f)

# Load llama translations
print("Loading llama translations...")
try:
    llama_df = pd.read_csv('llama_translations.csv', on_bad_lines='skip')
    print(f"  Loaded {len(llama_df)} llama translation rows")
except Exception as e:
    print(f"Error loading llama translations: {e}")
    llama_df = pd.DataFrame(columns=['triplet_id', 'variant_type', 'llama_translation'])

# Load gemma translations
print("Loading gemma translations...")
try:
    # Try with different quote handling
    gemma_df = pd.read_csv('gemma_translations.csv', on_bad_lines='skip', quoting=3, engine='python')
    print(f"  Loaded {len(gemma_df)} gemma translation rows")
except Exception as e:
    print(f"Error loading gemma translations with quoting=3: {e}")
    try:
        # Fallback: read line by line
        import csv
        rows = []
        with open('gemma_translations.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    rows.append(row)
                except Exception as row_error:
                    print(f"  Skipping row {i+2}: {row_error}")
        gemma_df = pd.DataFrame(rows)
        print(f"  Loaded {len(gemma_df)} gemma translation rows using fallback method")
    except Exception as e2:
        print(f"Error with fallback method: {e2}")
        gemma_df = pd.DataFrame(columns=['triplet_id', 'variant_type', 'gemma_translation'])

# Create dictionaries for fast lookup
# Key: (triplet_id, variant_type) -> translation
llama_dict = {}
for _, row in llama_df.iterrows():
    key = (row['triplet_id'], row['variant_type'])
    llama_dict[key] = row['llama_translation']

gemma_dict = {}
for _, row in gemma_df.iterrows():
    key = (row['triplet_id'], row['variant_type'])
    gemma_dict[key] = row['gemma_translation']

# Merge translations into base dataset
print("Merging translations...")
for item in base_data:
    triplet_id = item['id']
    
    # Add translations for base variant
    item['base_llama_translation'] = llama_dict.get((triplet_id, 'base'), '')
    item['base_gemma_translation'] = gemma_dict.get((triplet_id, 'base'), '')
    
    # Add translations for topic_fronting variant
    item['topic_fronting_llama_translation'] = llama_dict.get((triplet_id, 'topic_fronting'), '')
    item['topic_fronting_gemma_translation'] = gemma_dict.get((triplet_id, 'topic_fronting'), '')
    
    # Add translations for emphasis_shift variant
    item['emphasis_shift_llama_translation'] = llama_dict.get((triplet_id, 'emphasis_shift'), '')
    item['emphasis_shift_gemma_translation'] = gemma_dict.get((triplet_id, 'emphasis_shift'), '')

# Save merged data to new JSON file
output_file = 'dataset/db_with_all_translations.json'
print(f"Saving merged data to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(base_data, f, indent=2, ensure_ascii=False)

print(f"Done! Processed {len(base_data)} entries.")
print(f"Output saved to: {output_file}")

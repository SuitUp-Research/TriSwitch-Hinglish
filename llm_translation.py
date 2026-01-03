# Install required packages if needed
# pip install openai pandas

from openai import OpenAI
import pandas as pd
import json

# ## 1. Configure NVIDIA API

api_key = "YOUR_NVIDIA_API_KEY_HERE" # Replace with your actual API key
base_url = "https://integrate.api.nvidia.com/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

model_name = "meta/llama-3.1-8b-instruct"
print(f"Using model: {model_name}")



# ## 2. Define Translation Function

def translate_with_llama(texts):
    outputs = []

    for text in texts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful translator. Translate Hinglish sentences to fluent English. Only provide the English translation, nothing else."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this Hinglish sentence to English: {text}"
                    }
                ],
                temperature=0.2,
                max_tokens=100
            )
            
            translation = response.choices[0].message.content.strip()
            outputs.append(translation)
        except Exception as e:
            print(f"Error translating '{text}': {e}")
            outputs.append("")

    return outputs


# ## 3. Load Dataset from JSON

# Load dataset from JSON file
with open('dataset/db_with_reference_en.json', 'r', encoding='utf-8') as f:
    sentence_triplets = json.load(f)

print(f"Loaded {len(sentence_triplets)} sentence triplets from dataset")



# ## 4. Translate All Variants

results = []

for triplet in sentence_triplets:
    triplet_id = triplet["id"]
    
    # Translate each variant
    texts = [
        triplet["base"],
        triplet["variant_topic_fronting"],
        triplet["variant_emphasis_shift"],
    ]

    translations = translate_with_llama(texts)

    base_trans, topic_trans, emphasis_trans = translations

    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "base",
        "input_hinglish": triplet["base"],
        "llama_translation": base_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "topic_fronting",
        "input_hinglish": triplet["variant_topic_fronting"],
        "llama_translation": topic_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "emphasis_shift",
        "input_hinglish": triplet["variant_emphasis_shift"],
        "llama_translation": emphasis_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    if triplet_id % 100 == 0:
        print(f"Completed triplet {triplet_id}")

print(f"\nTranslation complete! Generated {len(results)} translations.")

# ## 5. Save Results

# Create DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
output_file = "llama_translations.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(f"Total results: {len(df_results)} rows")
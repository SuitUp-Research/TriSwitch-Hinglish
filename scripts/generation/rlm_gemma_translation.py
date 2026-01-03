# Install required packages if needed
# &pip install transformers torch sentencepiece

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import json


# ## 1. Load RLM Model

model_name = "rudrashah/RLM-hinglish-translator"
print(f"Loading model: {model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

print(f"Model loaded on {device}")



# ## 2. Define Translation Function

def translate_with_gemma(texts, max_new_tokens=64):
    outputs = []

    for text in texts:
        prompt = (
            "Translate the following Hinglish sentence to fluent English.\n"
            f"Hinglish: {text}\n"
            "English:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Extract only the English part
        if "English:" in decoded:
            decoded = decoded.split("English:", 1)[1].strip()

        outputs.append(decoded)

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

    translations = translate_with_gemma(texts)

    base_trans, topic_trans, emphasis_trans = translations

    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "base",
        "input_hinglish": triplet["base"],
        "gemma_translation": base_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "topic_fronting",
        "input_hinglish": triplet["variant_topic_fronting"],
        "gemma_translation": topic_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    results.append({
        "triplet_id": triplet_id,
        "variant_type": "emphasis_shift",
        "input_hinglish": triplet["variant_emphasis_shift"],
        "gemma_translation": emphasis_trans,
        "reference_en": triplet.get("reference_en", "")
    })
    
    if triplet_id % 100 == 0:
        print(f"Completed triplet {triplet_id}")

print(f"\nTranslation complete! Generated {len(results)} translations.")

# ## 5. Save Results

# Create DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
output_file = "gemma_translations.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(f"Total results: {len(df_results)} rows")
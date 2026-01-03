import json
import csv
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "ar5entum/marianMT_hin_eng_cs"
DATASET_PATH = "dataset\\db.json"
OUTPUT_CSV = "marian_triswitch_outputs.csv"

# Load model
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()

def marian_prompt(sentence):
    return f"""Hindi-English code-switched sentence:

{sentence}
"""

# Load dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

rows = []

for item in dataset:
    sent_id = item["id"]
    base = item["base"]

    prompt = marian_prompt(base)
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        early_stopping=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    rows.append({
        "id": sent_id,
        "input_base": base,
        "model_output": decoded,
        "domain": item["domain"],
        "pattern": item["pattern"]
    })

    if sent_id % 25 == 0:
        print(f"Processed {sent_id} samples")

# Save results
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "input_base", "model_output", "domain", "pattern"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved outputs to {OUTPUT_CSV}")

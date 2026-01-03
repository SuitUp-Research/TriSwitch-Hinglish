import sys
from pathlib import Path
import pandas as pd
from bert_score import score


def main():
    csv_path = Path(__file__).resolve().parent.parent / "final_results" /"marian_triswitch_outputs.csv"
    if not csv_path.exists():
        print(f"ERROR: could not find CSV at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    preds = df["model_output"].astype(str).tolist()
    refs = df["input_base"].astype(str).tolist()

    P, R, F1 = score(
        preds,
        refs,
        lang="en", 
        verbose=True,
    )

    print("BERTScore F1 (avg):", F1.mean().item())


if __name__ == "__main__":
    main()
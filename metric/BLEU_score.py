import sys
from pathlib import Path
import pandas as pd
from sacrebleu import corpus_bleu


def main():
	csv_path = Path(__file__).resolve().parent.parent / "marian_triswitch_outputs.csv"
	if not csv_path.exists():
		print(f"ERROR: could not find CSV at {csv_path}")
		sys.exit(1)

	df = pd.read_csv(csv_path)

	# sacrebleu.corpus_bleu expects hypotheses (list[str]) and
	# references as a list of reference streams: [refs_list]
	refs_list = df["input_base"].astype(str).tolist()
	hypotheses = df["model_output"].astype(str).tolist()

	references = [refs_list]

	bleu = corpus_bleu(hypotheses, references)
	print("BLEU score:", bleu.score)


if __name__ == "__main__":
	main()
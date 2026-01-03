import json
import time
from pathlib import Path
from typing import Any, Dict, List
import asyncio

from googletrans import Translator


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def translate_base_to_reference_en(input_path: Path, output_path: Path) -> None:
    data = load_json(input_path)
    translator = Translator(service_urls=["translate.googleapis.com"])

    translated: List[Dict[str, Any]] = []

    for row in data:
        base = row.get("base")

        if not base or row.get("reference_en"):
            translated.append(row)
            continue

        try:
            result = await translator.translate(base, src="auto", dest="en")
            row["reference_en"] = result.text
        except Exception as exc:
            row["reference_en"] = None
            row["reference_en_error"] = str(exc)

        await asyncio.sleep(0.15)
        translated.append(row)

    save_json(output_path, translated)


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "dataset" / "db.json"
    output_path = root / "dataset" / "db_with_reference_en.json"

    asyncio.run(translate_base_to_reference_en(input_path, output_path))
    print(f"Wrote translated dataset to {output_path}")


if __name__ == "__main__":
    main()

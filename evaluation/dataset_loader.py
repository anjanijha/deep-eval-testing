import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        return json.load(file)

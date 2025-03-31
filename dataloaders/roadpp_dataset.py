# waymo_json_dataloader.py

import json
import os
from typing import Generator, Any, Dict

class WaymoRoadDataset:
    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        self.json_path = json_path

    def load_full(self) -> Dict[str, Any]:
        """Load the entire JSON file into memory."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data

    def stream_data(self) -> Generator[Dict[str, Any], None, None]:
        """Stream JSON entries one by one (for large files)."""
        with open(self.json_path, 'r') as f:
            # Handles large JSON arrays like: [ {...}, {...}, ... ]
            buffer = ""
            inside_object = False
            for line in f:
                line = line.strip()
                if line.startswith("[") or line.startswith("]"):
                    continue
                if line.startswith("{"):
                    inside_object = True
                    buffer = line
                elif inside_object:
                    buffer += line

                if line.endswith("},") or line.endswith("}"):
                    try:
                        yield json.loads(buffer.rstrip(","))
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                    buffer = ""
                    inside_object = False


if __name__ == "__main__":
    json_path = "./road_waymo_trainval_v1.0.json"  # Update with your actual path
    loader = WaymoRoadDataset(json_path)

    print("Loading and printing the first 3 entries...")
    for i, entry in enumerate(loader.stream_data()):
        print(json.dumps(entry, indent=2))
        if i >= 2:
            break
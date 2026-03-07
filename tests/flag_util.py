
import argparse
import json
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=128)
    parser.add_argument("--train_type", type=str, default="not_found")
    return parser.parse_args()
    
def get_config(train_type: str):
    config_path = Path(__file__).resolve().parents[1] / "scripts" / f"config_{train_type}.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
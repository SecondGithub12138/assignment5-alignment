
import argparse
import json
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=128)
    parser.add_argument("--train_type", type=str, default="not_found")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID for checkpoint naming in sweep")
    parser.add_argument("--config_path", type=str, default=None, help="Override config file path for sweep")
    return parser.parse_args()
    
def get_config(train_type: str, path_override: str | Path | None = None):
    if path_override is not None:
        config_path = Path(path_override)
    else:
        config_path = Path(__file__).resolve().parents[1] / "scripts" / f"config_{train_type}.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DATA_SET_SIZE = 1024

SCRIPTS = Path(__file__).resolve().parent
log_path = SCRIPTS.parent / "results" / f"run_ei_{DATA_SET_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Logging to: {log_path}")

with open(log_path, "a") as f:
    subprocess.run([sys.executable, str(SCRIPTS / "run_ei_experiment.py"), "--data_size", str(DATA_SET_SIZE)], stdout=f, stderr=f, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "run_ei_eval.py"), "--data_size", str(DATA_SET_SIZE)], stdout=f, stderr=f, check=True)

print(f"Done. Log: {log_path}")

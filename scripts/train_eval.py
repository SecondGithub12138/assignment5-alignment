import subprocess
import sys
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.flag_util import get_args

DATA_SET_SIZE = 16 # for sft and ei only
TRAIN_TYPE = get_args().train_type
if TRAIN_TYPE == "grpo":
    DATA_SET_SIZE = 2

SCRIPTS = Path(__file__).resolve().parent
log_path = SCRIPTS.parent / "results" / f"run_{TRAIN_TYPE}_{DATA_SET_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Logging to: {log_path}")

with open(log_path, "a") as f:
    subprocess.run([sys.executable, str(SCRIPTS / f"run_{TRAIN_TYPE}_experiment.py"), "--data_size", str(DATA_SET_SIZE)], stdout=f, stderr=f, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "run_eval.py"), "--data_size", str(DATA_SET_SIZE), "--train_type", str(TRAIN_TYPE)], stdout=f, stderr=f, check=True)

print(f"Done. Log: {log_path}")

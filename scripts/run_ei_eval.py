import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.eval_util import eval
from tests.flag_util import get_args

if __name__== "__main__":
    DATA_SET_SIZE = get_args().data_size
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/ei_{DATA_SET_SIZE}"
    test_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/test.jsonl"
    eval(test_data_path, model_save_path)
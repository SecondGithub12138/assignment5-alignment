import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_alignment.wandb_util import init_wandb_run, log_wandb_metrics, make_eval_metrics
from tests.eval_util import eval
from tests.flag_util import get_args

if __name__== "__main__":
    args = get_args()
    DATA_SET_SIZE = args.data_size
    TRAIN_TYPE = args.train_type
    ckpt_suffix = args.exp_id if args.exp_id else str(DATA_SET_SIZE)
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/{TRAIN_TYPE}_{ckpt_suffix}"
    test_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/test.jsonl"
    run = init_wandb_run(
        train_type=TRAIN_TYPE,
        exp_id=args.exp_id,
        data_size=DATA_SET_SIZE,
        job_type="eval",
        config={
            "checkpoint_path": model_save_path,
            "validation_step": args.validation_step,
        },
        run_suffix=os.environ.get("WANDB_RUN_SUFFIX"),
    )
    summary = eval(test_data_path, model_save_path)
    log_wandb_metrics(
        make_eval_metrics(
            accuracy=float(summary["accuracy"]),
            total=int(summary["total"]),
            correct_format_correct_answer=int(summary["correct_format_correct_answer"]),
            correct_format_wrong_answer=int(summary["correct_format_wrong_answer"]),
            wrong_format_wrong_answer=int(summary["wrong_format_wrong_answer"]),
            vllm_load_s=float(summary["vllm_load_s"]),
            inference_s=float(summary["inference_s"]),
            prompts_per_second=float(summary["prompts_per_second"]),
            validation_step=args.validation_step,
        )
    )
    run.finish()

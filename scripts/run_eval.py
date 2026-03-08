import sys
import json
from pathlib import Path
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.eval_util import eval
from tests.flag_util import get_args

if __name__== "__main__":
    args = get_args()
    DATA_SET_SIZE = args.data_size
    TRAIN_TYPE = args.train_type
    ckpt_suffix = args.exp_id if args.exp_id else str(DATA_SET_SIZE)
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/{TRAIN_TYPE}_{ckpt_suffix}"
    test_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/test.jsonl"
    metrics = eval(test_data_path, model_save_path)

    wandb_info_path = Path(model_save_path) / "wandb_info.json"
    wandb_run = None
    log_step = None
    eval_grpo_step = None
    if wandb_info_path.exists():
        with open(wandb_info_path) as f:
            wandb_info = json.load(f)
        log_step = wandb_info.get("global_grad_step")
        eval_grpo_step = wandb_info.get("grpo_step")
        wandb_run = wandb.init(
            project=wandb_info.get("project"),
            entity=wandb_info.get("entity"),
            id=wandb_info.get("run_id"),
            name=wandb_info.get("name"),
            group=wandb_info.get("group"),
            job_type="eval",
            resume="allow",
        )
    else:
        wandb_run = wandb.init(
            project="assignment5-alignment",
            entity=None,
            name=f"{TRAIN_TYPE}_{ckpt_suffix}_eval",
            job_type="eval",
        )

    if wandb_run is not None:
        log_data = {
            "eval/accuracy": metrics["accuracy"],
            "eval/answer_reward_mean": metrics["answer_reward_mean"],
            "eval/format_reward_mean": metrics["format_reward_mean"],
            "eval/correct_format_correct_answer": metrics["correct_format_correct_answer"],
            "eval/correct_format_wrong_answer": metrics["correct_format_wrong_answer"],
            "eval/wrong_format_wrong_answer": metrics["wrong_format_wrong_answer"],
            "eval/total": metrics["total"],
            "eval/vllm_load_s": metrics["vllm_load_s"],
            "eval/infer_s": metrics["eval_infer_s"],
            "eval/results_path": metrics["results_path"],
        }
        if eval_grpo_step is not None:
            log_data["eval/grpo_step"] = eval_grpo_step
        if log_step is not None:
            wandb_run.log(log_data, step=log_step)
        else:
            wandb_run.log(log_data)
        wandb_run.finish()

from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
import json 
import logging
import time

logger = logging.getLogger(__name__)

def eval(dataset_path: str, model_path: str):
    logging.basicConfig(level=logging.INFO)
    examples = load_dataset(dataset_path)
    # acquire template 
    template_path = "/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path) as f:
        r1zero_template = f.read().strip()
    # load model first 
    # read all the questions and r1zero template it and aggregate it
    # preparing eval_sampling_params
    # generate it through model 
    
    vllm_load_start = time.time()
    model = LLM(model = model_path)
    vllm_load_elapsed = time.time() - vllm_load_start
    logger.info(f"[vLLM load] {vllm_load_elapsed:.1f}s")
    prompts = []
    ground_truths = []
    answers = []
    for example in examples:
        prompts.append(r1zero_template.format(question=example["question"]))
        ground_truths.append(example["answer"].split("####")[-1])
        answers.append(example["answer"].split("####")[0])
    
    eval_sampling_params = SamplingParams(
        temperature = 1.0,
        top_p = 0.9,
        max_tokens = 1024,
    )
    infer_start = time.time()
    raws = model.generate(prompts, eval_sampling_params)
    infer_elapsed = time.time() - infer_start
    logger.info(f"[vLLM inference] {infer_elapsed:.1f}s for {len(prompts)} prompts ({len(prompts)/infer_elapsed:.1f} prompts/s)")
    # following step is evaluation 
    # build response, ground truth
    # eval
    # write out
    all_metrics = []
    model_name = Path(model_path).name
    output_path = f"/home/seanlinux/assignment5-alignment/results/{model_name}_results.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for prompt, raw, ground_truth, answer in zip (prompts, raws, ground_truths, answers):
            response = raw.outputs[0].text 
            metrics = r1_zero_reward_fn(response, ground_truth, False)
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                "answer": answer,
                "metrics": metrics
            }
            f.write(json.dumps(result) + "\n")
            all_metrics.append(metrics)
    # (1) correct with both format and answer reward 1, 
    # (2) format reward 1 and answer reward 0, 
    # (3) format reward 0 and answer reward 0
    total = len(all_metrics)
    correct_format_correct_answer = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 1.0)
    correct_format_wrong_answer = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 0.0)
    wrong_format_wrong_answer = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 0.0)
    accuracy = correct_format_correct_answer / total if total > 0 else 0.0
    answer_reward_mean = sum(m["answer_reward"] for m in all_metrics) / total if total > 0 else 0.0
    format_reward_mean = sum(m["format_reward"] for m in all_metrics) / total if total > 0 else 0.0

    logger.info(f"Format correct, answer correct : {correct_format_correct_answer} ({accuracy:.1%})")
    logger.info(f"Format correct, answer wrong   : {correct_format_wrong_answer} ({correct_format_wrong_answer/total:.1%})")
    logger.info(f"Format wrong, answer wrong     : {wrong_format_wrong_answer} ({wrong_format_wrong_answer/total:.1%})")
    logger.info(f"Total                          : {total}")

    del model
    import gc
    gc.collect()
    return {
        "accuracy": accuracy,
        "answer_reward_mean": answer_reward_mean,
        "format_reward_mean": format_reward_mean,
        "correct_format_correct_answer": correct_format_correct_answer,
        "correct_format_wrong_answer": correct_format_wrong_answer,
        "wrong_format_wrong_answer": wrong_format_wrong_answer,
        "total": total,
        "vllm_load_s": vllm_load_elapsed,
        "eval_infer_s": infer_elapsed,
        "results_path": output_path,
    }

def load_dataset(dataset_path) -> list[dict]:
    examples = []
    with open(dataset_path) as f:
        for line in f:
            example  = json.loads(line)
            examples.append(example)
    return examples
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from vllm import LLM, SamplingParams
import json 
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    # parse dataset as list of dict: {question: foo; answer: bar}
    dataset_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/train.jsonl"
    examples = load_dataset(dataset_path)
    # acquire template 
    template_path = "/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path) as f:
        r1zero_template = f.read().strip()
    # load model first 
    # read all the questions and r1zero template it and aggregate it
    # preparing eval_sampling_params
    # generate it through model 
    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    model = LLM(model = model_path)
    prompts = []
    ground_truths = []
    for example in examples:
        prompts.append(r1zero_template.format(question=example["question"]))
        ground_truths.append(example["answer"].split("####")[-1])
    
    eval_sampling_params = SamplingParams(
        temperature = 1.0,
        top_p = 0.9,
        max_tokens = 1024,
    )
    raws = model.generate(prompts, eval_sampling_params)
    # following step is evaluation 
    # build response, ground truth
    # eval
    # write out
    all_metrics = []
    output_path = "/home/seanlinux/assignment5-alignment/gsm8k_baseline_results_ai.jsonl"
    with open(output_path, "w") as f:
        for prompt, raw, ground_truth in zip (prompts, raws, ground_truths):
            response = raw.outputs[0].text 
            metrics = r1_zero_reward_fn(response, ground_truth, False)
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                "metrics": metrics
            }
            f.write(json.dumps(result) + "\n")
            all_metrics.append(metrics)
    # (1) correct with both format and answer reward 1, 
    # (2) format reward 1 and answer reward 0, 
    # (3) format reward 0 and answer reward 0
    correct_format_correct_answer = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 1.0)
    correct_format_wrong_answer = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 0.0)
    wrong_format_wrong_answer = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 0.0)
    logger.info(f"Format correct, answer correct : {correct_format_correct_answer}")
    logger.info(f"Format correct, answer wrong : {correct_format_wrong_answer}")
    logger.info(f"Format wrong, answer wrong : {wrong_format_wrong_answer}")
    logger.info(f"Total : {len(all_metrics)}")

    del model
    import gc
    gc.collect()

def load_dataset(dataset_path) -> list[dict]:
    examples = []
    with open(dataset_path) as f:
        for line in f:
            example  = json.loads(line)
            examples.append(example)
    return examples

if __name__ == "__main__":
    main()
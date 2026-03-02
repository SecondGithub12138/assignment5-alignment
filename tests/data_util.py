
import json
import random

def sample_batch(data_set_size: int):
    # Preparing data
    train_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/train.jsonl"
    print("[2/6] Loading training data ...")
    all_samples = []
    with open(train_data_path, "r") as f:
        for line in f:
            all_samples.append(json.loads(line))
    samples = random.sample(all_samples, data_set_size)
    template_path = "/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path, "r") as f:
        r1_zero_template = f.read()
    prompt_strs = []
    output_strs = []
    for sample in samples:
        prompt_strs.append(r1_zero_template.format(question=sample["question"]))
        reasoning_answer = sample["answer"].split("####")
        output_strs.append(" " + reasoning_answer[0] + " </think> <answer> " + reasoning_answer[1].strip() + " </answer>")
    print(f"      Loaded {len(samples)} samples (DATA_SET_SIZE={data_set_size}).")
    return prompt_strs, output_strs

def simple_sample_batch(data_set_size: int):
    # Preparing data
    train_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/train.jsonl"
    print("[2/6] Loading training data ...")
    samples = []
    with open(train_data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= data_set_size:  # only load what we need
                break
    template_path = "/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path, "r") as f:
        r1_zero_template = f.read()
    prompt_strs = []
    output_strs = []
    for sample in samples:
        prompt_strs.append(r1_zero_template.format(question=sample["question"]))
        reasoning_answer = sample["answer"].split("####")
        output_strs.append(" " + reasoning_answer[0] + " </think> <answer> " + reasoning_answer[1].strip() + " </answer>")
    print(f"      Loaded {len(samples)} samples (DATA_SET_SIZE={data_set_size}).")
    return prompt_strs, output_strs
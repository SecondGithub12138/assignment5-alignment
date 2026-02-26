"""
Goal: sft training
Process
1. load model Qwen2.5
2. load train & test data set from /home/seanlinux/assignment5-alignment/data/gsm8k as list [question: foo, answer: bar] | *todo: get ground truth, 
3. get {128, 256, 512, 1024} size of D_b, optional:shuffle the order of the sampels
4. prepare prompt and label data 
5. feed to model and train, batch by batch, from 0 ~ bath-1, ... and so on, optional: gradient accumulation only when batch size is too big
6. finish the training and save it as checkpoint (depends on how long does the training process takes, if not too long, this step can be avoided)
7. run against the test dataset and log out 

A conversation between User and Assistant. The User asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. 
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, 
respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>

{"question": "My kitchen floor has a total area of 200 SqFt. I want to install new square floor tiles that cost $12 each, 
and each tile side is 1ft in length. How much will it cost me to renovate my kitchen floor?", 
"answer": "The total area of a square tile is length * width or 1*1 = <<1*1=1>>1 SqFt\nIf the kitchen has a total area of 200 SqFt 
then my kitchen needs 200/1 = <<200/1=200>>200 square tiles\nIf each tile costs $12, then 200 square tiles will cost $12 * 200 = 
$<<12*200=2400>>2,400\n#### 2400"}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
from tests.eval_util import eval
import json

import torch



def main():
    # Preparing model
    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # Preparing data
    train_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/train.jsonl"
    test_data_path = "/home/seanlinux/assignment5-alignment/data/gsm8k/test.jsonl"
    samples = []
    with open(train_data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    template_path = "/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path, "r") as f:
        r1_zero_template = f.read()
    prompt_strs = []
    output_strs = []
    for sample in samples:
        prompt_strs.append(r1_zero_template.format(question=sample["question"]))
        reasoning_answer = sample["answer"].split("####")
        output_strs.append(" " + reasoning_answer[0] + " </think> <answer> " + reasoning_answer[1].strip() + " </answer>")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    # "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    #     the tokenized prompt and output strings, with the final token sliced off.
    # "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    #     shifted input_ids (i.e., the input_ids without the first token).
    # "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    #     a mask on the response tokens in `labels`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    cur_gradient_step = 0
    MICRO_BATCH = 8 
    GRAD_ACCUM = 4
    DATA_SET_SIZE = 128 # {128, 256, 512, 1024}
    for i in range(0, len(train_data["input_ids"]), MICRO_BATCH):
        log_probs = run_get_response_log_probs(model, train_data["input_ids"][i: i + MICRO_BATCH], train_data["labels"][i: i + MICRO_BATCH], True)
        # dict[str, torch.Tensor]:
        #     "log_probs": torch.Tensor of shape (batch_size, sequence_length):
        #         the conditional log-probs of the response given the prompt.
        #         Note that we have not masked out the token indices corresponding
        #         to the prompt or padding; that is done in the train loop.
        #     "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
        #         the entropy of the next token predictions. As with the log-probs,
        #         we have not masked out the token indices corresponding to the prompt
        #         or padding; that is done in the train loop.
        run_sft_microbatch_train_step(log_probs["log_probs"], train_data["response_mask"][i: i + MICRO_BATCH], gradient_accumulation_steps=GRAD_ACCUM, normalize_constant=1)
        cur_gradient_step += 1
        if cur_gradient_step % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
    model_save_path = "/home/seanlinux/assignment5-alignment/checkpoints/sft_128"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    eval(test_data_path, model_save_path)

if __name__== "__main__":
    main()

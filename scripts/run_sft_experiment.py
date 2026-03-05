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

from transformers import AutoTokenizer
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
from tests.sft_util import preparing_model, sft
from tests.data_util import sample_batch
from tests.flag_util import get_args, get_config
import torch
import time
import json

def main():
    DATA_SET_SIZE = get_args().data_size
    model_path = get_config("sft")["model_path"]
    model = preparing_model(model_path)
    
    prompt_strs, output_strs = sample_batch(DATA_SET_SIZE)
    print("[3/6] Tokenizing data ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sft(prompt_strs, output_strs, tokenizer, model, "sft")
    print("[5/6] Saving model ...")
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/sft_{DATA_SET_SIZE}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")

if __name__== "__main__":
    main()

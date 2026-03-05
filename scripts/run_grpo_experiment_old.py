import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from typing import Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.sft_util import preparing_model, sft
from tests.data_util import sample_batch
from transformers import AutoTokenizer
from tests.flag_util import get_args, get_config
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step, run_tokenize_prompt_and_output, run_get_response_log_probs

def main():
    # DATA_SET_SIZE = get_args().data_size
    config = get_config("grpo")
    n_grpo_steps: int = config["n_grpo_steps"]
    lr: float = config["lr"]
    advantage_eps: float =  config["advantage_eps"]
    
    group_size: int = config["rollouts"]
    sampling_temperature: float = config["sampling_temperature"]
    sampling_min_tokens: int = config["sampling_min_tokens"] # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = config["sampling_max_tokens"]
    grad_accum: int = config["grad_accum"] # microbatch size is 2, will fit on H100
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = config["loss_type"] # 'no_baseline', 'reinforce_with_baseline', 'grpo_clip'
    use_std_normalization: bool = config["use_std_normalization"]
    weight_decay = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    micro_batch_gen = config["micro_batch_gen"]
    train_batch_size = config["train_batch_size"] # for each step() 256
    assert train_batch_size % grad_accum == 0, ( # grad_accum 8
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_batch = train_batch_size // grad_accum # 256 / 8 = 32 ----------!!!
    assert micro_batch % group_size == 0, (
        "micro_batch must be divisible by group_size"
    )
    rollout_batch_size = config["rollout_batch_size"] # 256
    assert rollout_batch_size % group_size == 0, ( # rollout_batch_size 256 group_size 8
        "rollout_batch_size must be divisible by group_size"
    )
    DATA_SET_SIZE = rollout_batch_size // group_size
    # gpu_memory_utilization: float = 0.85
    # epochs_per_rollout_batch: int = 1 # On-policy
    
    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    model = preparing_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    optimizer.zero_grad()
    for _ in range(n_grpo_steps):
        model.gradient_checkpointing_disable()
        all_prompt_strs, all_output_strs = sample_batch(DATA_SET_SIZE)  # Sample D_b once per EI step
        all_repeated_prompt_strs = []
        all_repeated_ground_truths = []
        all_generated_sample_outputs = []
        for batch_idx in range(DATA_SET_SIZE//micro_batch_gen):
            start = batch_idx * micro_batch_gen
            prompt_strs = all_prompt_strs[start : start + micro_batch_gen]
            output_strs = all_output_strs[start : start + micro_batch_gen]
            
            repeated_prompt_strs = [p for p in prompt_strs for _ in range(group_size)]
            ground_truths = []
            for output_str in output_strs:
                ground_truths.append(output_str.split("<answer>")[-1].replace("</answer>", "").strip())
            repeated_ground_truths = [t for t in ground_truths for _ in range(group_size)]
            all_repeated_prompt_strs.extend(repeated_prompt_strs)
            all_repeated_ground_truths.extend(repeated_ground_truths)

            tokenizer.padding_side = "left"
            inputs = tokenizer(repeated_prompt_strs, return_tensors="pt", padding=True).to(model.device) # using hugging face, transfer str to id
            generated_ids = model.generate(**inputs, max_new_tokens=sampling_max_tokens, min_new_tokens=sampling_min_tokens, do_sample=True, temperature=sampling_temperature) # generate id
            all_generated_sample_outputs.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)) # transfer id to str
            
            del generated_ids
            torch.cuda.empty_cache()
        # Data generated, start trining
        model.gradient_checkpointing_enable()
        tokenizer.padding_side = "right" 
        device = torch.device("cuda:0")
        cur_gradient_step = 0
        for i in range(0, len(all_repeated_prompt_strs), micro_batch):
            micro_repeated_prompt_strs = all_repeated_prompt_strs[i: i + micro_batch]
            micro_generated_sample_outputs = all_generated_sample_outputs[i: i + micro_batch]
            micro_ground_truths = all_repeated_ground_truths[i: i + micro_batch]

            train_data = run_tokenize_prompt_and_output(micro_repeated_prompt_strs, micro_generated_sample_outputs, tokenizer)
            
            input_ids = train_data["input_ids"].to(device)
            labels = train_data["labels"].to(device)
            response_mask = train_data["response_mask"].to(device)

            log_probs = run_get_response_log_probs(model, input_ids, labels, False)["log_probs"]
            advantages, raw_rewards, _ = run_compute_group_normalized_rewards(r1_zero_reward_fn, micro_generated_sample_outputs, micro_ground_truths, group_size, advantage_eps, use_std_normalization)
            advantages = advantages.unsqueeze(-1).to(device)
            raw_rewards = raw_rewards.unsqueeze(-1).to(device)
            loss, _ = run_grpo_microbatch_train_step(log_probs, response_mask, grad_accum, loss_type, raw_rewards, advantages, None, 1)
            cur_gradient_step += 1
            if cur_gradient_step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
    # Data collection done, start sft
    print("Saving model ...")
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/grpo_{DATA_SET_SIZE}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")

if __name__ == "__main__":
    main()
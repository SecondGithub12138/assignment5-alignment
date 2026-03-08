import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from typing import Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.sft_util import preparing_model, sft
from tests.data_util import sample_batch
from transformers import AutoTokenizer
from tests.flag_util import get_args, get_config
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step, run_tokenize_prompt_and_output, run_get_response_log_probs, run_get_response_log_probs_chunked, run_masked_mean

def main():
    args = get_args()
    config = get_config("grpo", path_override=args.config_path)
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
    loss_aggregation: Literal["masked_mean", "masked_normalize"] = config.get("loss_aggregation", "masked_mean")
    weight_decay = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    micro_batch_gen = config["micro_batch_gen"]

    epochs_per_rollout_batch: int = config["epochs_per_rollout_batch"]
    rollout_batch_size = config["rollout_batch_size"] 
    train_batch_size = config["train_batch_size"]
    assert train_batch_size % grad_accum == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    
    assert rollout_batch_size % group_size == 0, ( # rollout_batch_size 256 group_size 8
        "rollout_batch_size must be divisible by group_size"
    )
    DATA_SET_SIZE = rollout_batch_size // group_size
    # gpu_memory_utilization: float = 0.85
    

    print(f"[Hyperparameters] n_grpo_steps={n_grpo_steps} | lr={lr} | grad_accum={grad_accum} | group_size={group_size} | rollout_batch_size={rollout_batch_size} | train_batch_size={train_batch_size} | DATA_SET_SIZE={DATA_SET_SIZE} | micro_batch_gen={micro_batch_gen} | loss_type={loss_type} | loss_aggregation={loss_aggregation} | use_std_normalization={use_std_normalization} | epochs_per_rollout_batch={epochs_per_rollout_batch} | sampling_temperature={sampling_temperature} | sampling_min_tokens={sampling_min_tokens} | sampling_max_tokens={sampling_max_tokens} | advantage_eps={advantage_eps} | weight_decay={weight_decay} | betas=({beta1},{beta2})")

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
    for grpo_step in range(n_grpo_steps):
        model.gradient_checkpointing_disable()
        all_prompt_strs, all_output_strs = sample_batch(DATA_SET_SIZE)  # Sample D_b once per EI step
        all_repeated_prompt_strs = []
        all_repeated_ground_truths = []
        all_generated_sample_outputs = []
        all_old_log_probs_list = []
        gen_start = time.time()
        for batch_gen_idx in range(DATA_SET_SIZE//micro_batch_gen):
            print(f"batch_gen_idx: {batch_gen_idx}")
            start = batch_gen_idx * micro_batch_gen
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
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=sampling_max_tokens, min_new_tokens=sampling_min_tokens, do_sample=True, temperature=sampling_temperature) # generate id
                prompt_len = inputs["input_ids"].shape[1]
                generated_sample_outputs = tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
                del generated_ids, inputs
                torch.cuda.empty_cache()
                old_train_data = run_tokenize_prompt_and_output(repeated_prompt_strs, generated_sample_outputs, tokenizer)
                old_log_probs = run_get_response_log_probs_chunked(
                    model, old_train_data["input_ids"], old_train_data["labels"],
                    return_token_entropy=False, chunk_size=min(micro_batch_gen, 16), device=model.device,
                )["log_probs"]
            print(f"      [sample output] {repr(generated_sample_outputs[0])}")
            all_generated_sample_outputs.extend(generated_sample_outputs) # transfer id to str
            all_old_log_probs_list.extend(old_log_probs)
            torch.cuda.empty_cache()
        print(f"Data generation done ... ({time.time() - gen_start:.1f}s)")
        # Data generated, start trining
        model.gradient_checkpointing_enable()
        tokenizer.padding_side = "right" 
        device = torch.device("cuda:0")
        all_old_log_probs = torch.stack(all_old_log_probs_list, dim=0).to(device)
        all_advantages, all_raw_rewards, _ = run_compute_group_normalized_rewards(r1_zero_reward_fn, all_generated_sample_outputs, all_repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        train_start = time.time()
        for _ in range(epochs_per_rollout_batch): # epochs
            # todo: could randomize the rollout_batch data 
            for train_batch_i in range(rollout_batch_size // train_batch_size): # how many train_batch per rollout batch
                micro_batch = train_batch_size // grad_accum
                for grad_accum_step in range(grad_accum): # how many round of gradient accumulation 
                    start = train_batch_i * train_batch_size + grad_accum_step * micro_batch
                    micro_repeated_prompt_strs = all_repeated_prompt_strs[start: start + micro_batch]
                    micro_generated_sample_outputs = all_generated_sample_outputs[start: start + micro_batch]
                    micro_old_log_probs = all_old_log_probs[start: start + micro_batch]
                    micro_advantages = all_advantages[start: start + micro_batch]
                    micro_raw_rewards = all_raw_rewards[start: start + micro_batch]
                    
                    train_data = run_tokenize_prompt_and_output(micro_repeated_prompt_strs, micro_generated_sample_outputs, tokenizer)
                    input_ids = train_data["input_ids"].to(device)
                    labels = train_data["labels"].to(device)
                    response_mask = train_data["response_mask"].to(device)
                    output = run_get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs = output["log_probs"]
                    token_entropy = output["token_entropy"]
                    micro_advantages = micro_advantages.unsqueeze(-1).to(device)
                    micro_raw_rewards = micro_raw_rewards.unsqueeze(-1).to(device)
                    loss, _ = run_grpo_microbatch_train_step(log_probs, response_mask, grad_accum, loss_type, micro_raw_rewards, micro_advantages, micro_old_log_probs, 1, loss_aggregation)
                    mean_entropy = run_masked_mean(token_entropy, response_mask).item()
                    mean_response_len = response_mask.sum(dim=-1).float().mean().item()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                    print(f"      [grad norm] {grad_norm:.4f}")
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"      [grad step {grad_accum_step} / {grad_accum}] loss={loss:.6f} entropy={mean_entropy:.4f} response_len={mean_response_len:.1f}  [mem peak] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")
                optimizer.step()
                optimizer.zero_grad()
        grpo_alloc = torch.cuda.memory_allocated() / 1024**3
        grpo_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"      [grpo_step {grpo_step}/{n_grpo_steps} finished] loss={loss:.6f} entropy={mean_entropy:.4f} response_len={mean_response_len:.1f} train_time={time.time()-train_start:.1f}s  [mem peak] allocated={grpo_alloc:.2f}GiB, reserved={grpo_reserved:.2f}GiB")
    # Data collection done, start sft
    print("Saving model ...")
    exp_id = args.exp_id or str(args.data_size)
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/grpo_{exp_id}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")

if __name__ == "__main__":
    main()
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from typing import Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.sft_util import preparing_model, sft
from tests.data_util import sample_batch
from transformers import AutoTokenizer, PreTrainedModel
from tests.flag_util import get_args, get_config
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step, run_tokenize_prompt_and_output, run_get_response_log_probs
from vllm import LLM, SamplingParams
from vllm.utils.torch_utils import set_random_seed
from unittest.mock import patch

def init_vllm(model: str, seed: int, gpu_memory_utilization: float):
    set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    with world_size_patch:
        return LLM(
            model = model,
            dtype = torch.bfloat16,
            enable_prefix_caching = True,
            gpu_memory_utilization = gpu_memory_utilization,
            data_parallel_size = 1,
        )

def load_policy_into_vllm_instance(llm: LLM, model: PreTrainedModel):
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    llm.apply_model(lambda m: m.load_state_dict(state_dict, strict=False))


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
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    rollout_batch_size = config["rollout_batch_size"] # 256
    assert rollout_batch_size % group_size == 0, ( # rollout_batch_size 256 group_size 8
        "rollout_batch_size must be divisible by group_size"
    )
    DATA_SET_SIZE = rollout_batch_size // group_size
    assert DATA_SET_SIZE % micro_batch_gen == 0 and DATA_SET_SIZE >= micro_batch_gen, ( 
        "micro_batch_gen can't be larger than DATA_SET_SIZE"
    )
    gpu_memory_utilization: float = config["gpu_memory_utilization"]
    # epochs_per_rollout_batch: int = 1 # On-policy

    print(f"[Hyperparameters] n_grpo_steps={n_grpo_steps} | lr={lr} | grad_accum={grad_accum} | micro_batch={micro_batch} | group_size={group_size} | rollout_batch_size={rollout_batch_size} | train_batch_size={train_batch_size} | DATA_SET_SIZE={DATA_SET_SIZE} | micro_batch_gen={micro_batch_gen} | loss_type={loss_type} | use_std_normalization={use_std_normalization} | sampling_temperature={sampling_temperature} | sampling_min_tokens={sampling_min_tokens} | sampling_max_tokens={sampling_max_tokens} | advantage_eps={advantage_eps} | weight_decay={weight_decay} | betas=({beta1},{beta2}) | gpu_memory_utilization={gpu_memory_utilization}")

    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    model = preparing_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    optimizer.zero_grad(set_to_none=True)
    device = torch.device("cuda:0")
    gen_model = init_vllm(model_path, 4, gpu_memory_utilization)
    samplingParams = SamplingParams(
        temperature = sampling_temperature,
        min_tokens = sampling_min_tokens,
        max_tokens = sampling_max_tokens
    )
    for grpo_step in range(n_grpo_steps):
        step_start = time.time()
        model.gradient_checkpointing_disable()
        all_prompt_strs, all_output_strs = sample_batch(DATA_SET_SIZE)  # Sample D_b once per EI step
        all_repeated_prompt_strs = []
        all_repeated_ground_truths = []
        all_generated_sample_outputs = []
        gen_start = time.time()
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
            raws = gen_model.generate(repeated_prompt_strs, samplingParams)
            generated_texts = [raw.outputs[0].text for raw in raws]
            all_generated_sample_outputs.extend(generated_texts)
            print(f"      [sample output] {repr(generated_texts[0])}")
        print(f"Data generation done ... ({time.time() - gen_start:.1f}s)")
        # Generation done, release vLLM KV cache memory before training
        gen_model.sleep()
        torch.cuda.empty_cache()
        # Data generated, start training
        model.gradient_checkpointing_enable()
        tokenizer.padding_side = "right" 
        
        cur_gradient_step = 0
        train_start = time.time()
        advantages, raw_rewards, _ = run_compute_group_normalized_rewards(r1_zero_reward_fn, all_generated_sample_outputs, all_repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        for i in range(0, len(all_repeated_prompt_strs), micro_batch):
            micro_repeated_prompt_strs = all_repeated_prompt_strs[i: i + micro_batch]
            micro_generated_sample_outputs = all_generated_sample_outputs[i: i + micro_batch]
            micro_ground_truths = all_repeated_ground_truths[i: i + micro_batch]
            micro_advantages = advantages[i: i + micro_batch]
            micro_raw_rewards = raw_rewards[i: i + micro_batch]
            train_data = run_tokenize_prompt_and_output(micro_repeated_prompt_strs, micro_generated_sample_outputs, tokenizer)
            input_ids = train_data["input_ids"].to(device)
            labels = train_data["labels"].to(device)
            response_mask = train_data["response_mask"].to(device)
            log_probs = run_get_response_log_probs(model, input_ids, labels, False)["log_probs"]
            print(f"      [DEBUG] raw_rewards: {micro_raw_rewards.tolist()}")
            print(f"      [DEBUG] advantages: {micro_advantages.tolist()}")
            print(f"      [DEBUG] response_mask nonzero count: {response_mask.sum().item()}")
            print(f"      [DEBUG] sample output[0]: {repr(micro_generated_sample_outputs[0][:2000])}")
            print(f"      [DEBUG] ground truth[0]: {repr(micro_ground_truths[0])}")
            micro_advantages = micro_advantages.unsqueeze(-1).to(device)
            micro_raw_rewards = micro_raw_rewards.unsqueeze(-1).to(device)
            loss, _ = run_grpo_microbatch_train_step(log_probs, response_mask, grad_accum, loss_type, micro_raw_rewards, micro_advantages, None, 1)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            print(f"      [grad norm] {grad_norm:.4f}")
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"      [grad step {cur_gradient_step}/{grad_accum}] loss={loss:.4f}  [mem peak] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")
            cur_gradient_step += 1
            if cur_gradient_step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # Training done, sync weights back to vLLM and wake it up for next rollout
        gen_model.wake_up()
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        gen_model.apply_model(lambda m: m.load_state_dict(state_dict, strict=False))
        now = time.time()
        total_time = now - step_start
        train_time = now - train_start
        gen_time = total_time - train_time
        grpo_alloc = torch.cuda.memory_allocated() / 1024**3
        grpo_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"      [grpo_step {grpo_step}/{n_grpo_steps} finished] loss={loss:.6f}  total={total_time:.1f}s (gen={gen_time:.1f}s train={train_time:.1f}s)  [mem peak] allocated={grpo_alloc:.2f}GiB, reserved={grpo_reserved:.2f}GiB")
    # Data collection done, start sft
    print("Saving model ...")
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/grpo_{get_args().data_size}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")

if __name__ == "__main__":
    main()
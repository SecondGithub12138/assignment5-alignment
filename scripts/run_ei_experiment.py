import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.sft_util import preparing_model, sft
from tests.data_util import generate_until_answer, sample_batch
from transformers import AutoTokenizer
from tests.flag_util import get_args, get_config

def main():
    DATA_SET_SIZE = get_args().data_size
    config = get_config("ei")
    micro_batch_gen = config["micro_batch_gen"] #Gen means generate, happened during data generation stage
    model_path = config["model_path"]
    ei_steps = config["ei_steps"]
    rollouts = config["rollouts"]
    sampling_max_tokens = config["sampling_max_tokens"]
    sampling_min_tokens = config["sampling_min_tokens"]
    sampling_temperature = config["sampling_temperature"]

    model = preparing_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for i in range(ei_steps):
        filtered_sample_prompts = []
        filtered_sample_outputs = []
        total_generated = 0
        model.gradient_checkpointing_disable()
        all_prompt_strs, all_output_strs = sample_batch(DATA_SET_SIZE)  # Sample D_b once per EI step
        for batch_idx in range(DATA_SET_SIZE//micro_batch_gen):
            start = batch_idx * micro_batch_gen
            prompt_strs = all_prompt_strs[start : start + micro_batch_gen]
            output_strs = all_output_strs[start : start + micro_batch_gen]
            tokenizer.padding_side = "left"
            ground_truths = []
            for output_str in output_strs:
                ground_truths.append(output_str.split("<answer>")[-1].replace("</answer>", "").strip())
            for _ in range(rollouts):
                inputs = tokenizer(prompt_strs, return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode():
                    prompt_len = inputs["input_ids"].shape[1]
                    generated_ids = generate_until_answer(
                        model,
                        tokenizer,
                        inputs,
                        prompt_input_len=prompt_len,
                        min_new_tokens=sampling_min_tokens,
                        max_new_tokens=sampling_max_tokens,
                        do_sample=True,
                        temperature=sampling_temperature,
                    )
                    generated_sample_outputs = tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
                total_generated += len(generated_sample_outputs)
                del generated_ids, inputs
                torch.cuda.empty_cache()
                # decided which output got right ground truth answer to keep
                for response, ground_truth, prompt in zip(generated_sample_outputs, ground_truths, prompt_strs):
                    if r1_zero_reward_fn(response, ground_truth, False)["reward"] == 0:
                        continue
                    filtered_sample_outputs.append(response)
                    filtered_sample_prompts.append(prompt)
        # Data collection done, start sft
        print(f"[EI step {i+1}/{ei_steps}] Collection done: generated={total_generated}, filtered={len(filtered_sample_prompts)}")
        if filtered_sample_prompts:
            model.gradient_checkpointing_enable()
            tokenizer.padding_side = "right" 
            sft(filtered_sample_prompts, filtered_sample_outputs, tokenizer, model, "ei")
    print("Saving model ...")
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/ei_{DATA_SET_SIZE}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")


if __name__ == "__main__":
    main()
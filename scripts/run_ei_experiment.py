import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.sft_util import preparing_model, sft
from tests.data_util import sample_batch
from transformers import AutoTokenizer
from tests.flag_util import get_args

def main():
    DATA_SET_SIZE = get_args().data_size
    GEN_MICRO_BATCH = 5
    
    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    model = preparing_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    EI_STEPS = 5
    ROLLOUTS = 10
    for i in range(EI_STEPS):
        filtered_sample_prompts = []
        filtered_sample_outputs = []
        total_generated = 0
        model.gradient_checkpointing_disable()
        all_prompt_strs, all_output_strs = sample_batch(DATA_SET_SIZE)  # Sample D_b once per EI step
        for batch_idx in range(DATA_SET_SIZE//GEN_MICRO_BATCH):
            start = batch_idx * GEN_MICRO_BATCH
            prompt_strs = all_prompt_strs[start : start + GEN_MICRO_BATCH]
            output_strs = all_output_strs[start : start + GEN_MICRO_BATCH]
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompt_strs, return_tensors="pt", padding=True).to(model.device)
            ground_truths = []
            for output_str in output_strs:
                ground_truths.append(output_str.split("<answer>")[-1].replace("</answer>", "").strip())
            for _ in range(ROLLOUTS):
                generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8)
                generated_sample_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                total_generated += len(generated_sample_outputs)
                del generated_ids
                torch.cuda.empty_cache()
                # decided which output got right ground truth answer to keep
                for response, ground_truth, prompt in zip(generated_sample_outputs, ground_truths, prompt_strs):
                    if r1_zero_reward_fn(response, ground_truth, False)["reward"] == 0:
                        continue
                    filtered_sample_outputs.append(response)
                    filtered_sample_prompts.append(prompt)
            # Data collection done, start sft
        print(f"[EI step {i+1}/{EI_STEPS}] Collection done: generated={total_generated}, filtered={len(filtered_sample_prompts)}")
        if filtered_sample_prompts:
            model.gradient_checkpointing_enable()
            tokenizer.padding_side = "right" 
            sft(filtered_sample_prompts, filtered_sample_outputs, tokenizer, model)
    print("Saving model ...")
    model_save_path = f"/home/seanlinux/assignment5-alignment/checkpoints/ei_{DATA_SET_SIZE}"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"      Saved to {model_save_path}")
    print("[6/6] Training done. Run eval separately:")


if __name__ == "__main__":
    main()
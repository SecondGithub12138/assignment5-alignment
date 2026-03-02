import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
import torch
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
import time

def preparing_model(model_path: str) -> PreTrainedModel:
    # Preparing model
    print(f"[1/6] Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    print(f"      Model loaded. Device map: {model.hf_device_map}")
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"      [mem after model load] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")
    # model.gradient_checkpointing_enable()
    return model


def sft(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):
    MICRO_BATCH = 5
    DATA_SET_SIZE = len(prompt_strs)
    GRAD_ACCUM = 8  # keep total effective batch = 32
    train_data = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    print(f"      input_ids shape: {train_data['input_ids'].shape}")
    print("[4/6] Starting training ...")
    train_start = time.time()
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    cur_gradient_step = 0
    for i in range(0, DATA_SET_SIZE, MICRO_BATCH):
        print(f"      microbatch {i}~{i+MICRO_BATCH} / {DATA_SET_SIZE} ...", flush=True)
        input_ids = train_data["input_ids"][i: i + MICRO_BATCH].to(device)
        labels = train_data["labels"][i: i + MICRO_BATCH].to(device)
        response_mask = train_data["response_mask"][i: i + MICRO_BATCH].to(device)
        log_probs = run_get_response_log_probs(model, input_ids, labels, False)  # no entropy during train to save VRAM
        loss, _ = run_sft_microbatch_train_step(log_probs["log_probs"], response_mask, gradient_accumulation_steps=GRAD_ACCUM, normalize_constant=1)
        cur_gradient_step += 1
        if cur_gradient_step % GRAD_ACCUM == 0:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            optimizer.step()
            optimizer.zero_grad()
            print(f"      [grad step {cur_gradient_step // GRAD_ACCUM}] loss={loss:.4f}  [mem peak] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")

    train_elapsed = time.time() - train_start
    print(f"      Training finished in {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")
    
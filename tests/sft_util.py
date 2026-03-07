import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
import torch
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
from tests.flag_util import get_config
import time

def preparing_model(model_path: str) -> PreTrainedModel:
    # Preparing model
    print(f"[1/6] Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    print(f"      Model loaded. Device map: {model.hf_device_map}")
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"      [mem after model load] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")
    # model.gradient_checkpointing_enable()
    return model


def sft(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, train_type: str):
    config = get_config(train_type)
    micro_batch = config["micro_batch"]
    grad_accum = config["grad_accum"]  # keep total effective batch = 32
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]

    print("[4/6] Starting training ...")
    train_start = time.time()
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    optimizer.zero_grad()
    cur_gradient_step = 0
    for i in range(0, len(prompt_strs), micro_batch):
        micro_prompt_strs = prompt_strs[i: i + micro_batch]
        micro_output_strs = output_strs[i: i + micro_batch]
        train_data = run_tokenize_prompt_and_output(micro_prompt_strs, micro_output_strs, tokenizer)
        print(f"      microbatch {i}~{i+micro_batch} / {len(prompt_strs)} ...", flush=True)
        input_ids = train_data["input_ids"].to(device)
        labels = train_data["labels"].to(device)
        response_mask = train_data["response_mask"].to(device)
        output = run_get_response_log_probs(model, input_ids, labels, False)  # no entropy during train to save VRAM
        loss, _ = run_sft_microbatch_train_step(output["log_probs"], response_mask, gradient_accumulation_steps=grad_accum, normalize_constant=1)
        cur_gradient_step += 1
        if cur_gradient_step % grad_accum == 0:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            optimizer.step()
            optimizer.zero_grad()
            print(f"      [grad step {cur_gradient_step // grad_accum}] loss={loss:.4f}  [mem peak] allocated={alloc:.2f}GiB, reserved={reserved:.2f}GiB")

    train_elapsed = time.time() - train_start
    print(f"      Training finished in {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")
    
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make `tests` importable when running this script directly (not via pytest/package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.adapters import (
    log_generations,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
    run_tokenize_prompt_and_output,
)


@dataclass
class SFTExample:
    question: str
    prompt: str
    output: str
    ground_truth: str


def _extract_ground_truth(raw_answer: str) -> str:
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


def _load_prompt_template(path: Path) -> str:
    with path.open() as f:
        return f.read().strip()


def _load_math_sft(path: Path, prompt_template: str) -> list[SFTExample]:
    examples: list[SFTExample] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            question = ex.get("question") or ex.get("problem") or ex.get("prompt")
            if question is None:
                continue

            # If the dataset already stores a full prompt, use it as-is.
            if ex.get("prompt") and "{question}" not in ex.get("prompt", ""):
                prompt = ex["prompt"]
            else:
                prompt = prompt_template.format(question=question)

            output = ex.get("response") or ex.get("output") or ex.get("solution") or ex.get("completion")
            answer = ex.get("ground_truth") or ex.get("answer") or output
            if output is None or answer is None:
                continue

            examples.append(
                SFTExample(
                    question=question,
                    prompt=prompt,
                    output=output,
                    ground_truth=_extract_ground_truth(str(answer)),
                )
            )
    return examples


def _unique_by_question(examples: list[SFTExample]) -> list[SFTExample]:
    seen: set[str] = set()
    unique: list[SFTExample] = []
    for ex in examples:
        if ex.question in seen:
            continue
        seen.add(ex.question)
        unique.append(ex)
    return unique


def _split_train_val(examples: list[SFTExample], val_ratio: float, seed: int) -> tuple[list[SFTExample], list[SFTExample]]:
    rng = random.Random(seed)
    items = examples[:]
    rng.shuffle(items)
    val_size = max(1, int(len(items) * val_ratio))
    return items[val_size:], items[:val_size]


def _iter_batches(items: list[SFTExample], batch_size: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch = [items[j] for j in idx[i : i + batch_size]]
        yield batch


def train_sft(
    model: torch.nn.Module,
    tokenizer,
    train_examples: list[SFTExample],
    lr: float,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    global_step = 0
    running_loss = 0.0
    running_updates = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        pbar = tqdm(
            _iter_batches(train_examples, batch_size=batch_size, seed=seed + epoch),
            total=math.ceil(len(train_examples) / batch_size),
            desc=f"train epoch {epoch+1}/{epochs}",
        )
        for batch in pbar:
            prompts = [x.prompt for x in batch]
            outputs = [x.output for x in batch]
            toks = run_tokenize_prompt_and_output(prompts, outputs, tokenizer)
            input_ids = toks["input_ids"].to(device)
            labels = toks["labels"].to(device)
            response_mask = toks["response_mask"].to(device)

            token_log_probs = run_get_response_log_probs(
                model=model, input_ids=input_ids, labels=labels, return_token_entropy=False
            )["log_probs"]
            loss, _ = run_sft_microbatch_train_step(
                policy_log_probs=token_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            running_loss += float(loss.detach().item())
            global_step += 1

            if global_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                running_updates += 1
                pbar.set_postfix(loss=f"{running_loss / max(1, running_updates):.4f}")

    # Flush leftover gradients if needed.
    if global_step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {"train_loss_avg": running_loss / max(1, global_step)}


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    tokenizer,
    val_examples: list[SFTExample],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()

    all_prompts: list[str] = []
    all_responses: list[str] = []
    all_ground_truths: list[str] = []
    all_reward_infos: list[dict[str, float]] = []

    for i in tqdm(range(0, len(val_examples), batch_size), desc="eval"):
        chunk = val_examples[i : i + batch_size]
        prompts = [x.prompt for x in chunk]
        gts = [x.ground_truth for x in chunk]

        tok = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=-1).tolist()

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

        for row, prompt_len, gt, prompt in zip(gen_ids, prompt_lens, gts, prompts):
            response_ids = row[int(prompt_len) :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            reward_info = r1_zero_reward_fn(response, gt, fast=True)
            all_prompts.append(prompt)
            all_responses.append(response)
            all_ground_truths.append(gt)
            all_reward_infos.append(reward_info)

    answer_acc = sum(r.get("answer_reward", 0.0) for r in all_reward_infos) / max(1, len(all_reward_infos))
    logs = log_generations(
        prompt_strs=all_prompts,
        generated_responses=all_responses,
        ground_truths=all_ground_truths,
        reward_infos=all_reward_infos,
    )
    model.train()
    return float(answer_acc), logs


def filter_correct_examples(examples: list[SFTExample]) -> list[SFTExample]:
    filtered: list[SFTExample] = []
    for ex in tqdm(examples, desc="filter-correct"):
        reward_info = r1_zero_reward_fn(ex.output, ex.ground_truth, fast=True)
        if reward_info.get("answer_reward", 0.0) >= 1.0:
            filtered.append(ex)
    return filtered


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_template = _load_prompt_template(Path(args.prompt_template_path))
    all_examples = _load_math_sft(Path(args.dataset_path), prompt_template)
    all_examples = _unique_by_question(all_examples)

    train_examples, val_examples = _split_train_val(all_examples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Loaded {len(all_examples)} examples => train={len(train_examples)} val={len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    subset_sizes: list[int] = [128, 256, 512, 1024]
    subset_sizes = [s for s in subset_sizes if s <= len(train_examples)]
    subset_sizes.append(len(train_examples))

    curves: list[dict[str, Any]] = []
    for size in subset_sizes:
        print(f"\n=== SFT experiment size={size} ===")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        subset = train_examples[:size]
        train_stats = train_sft(
            model=model,
            tokenizer=tokenizer,
            train_examples=subset,
            lr=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
            seed=args.seed,
        )
        val_acc, logs = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            val_examples=val_examples,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            device=device,
        )
        curves.append(
            {
                "train_size": size,
                "val_answer_accuracy": val_acc,
                "train_stats": train_stats,
                "val_summary": logs["summary"],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Filtered full-data experiment ===")
    filtered_train = filter_correct_examples(train_examples)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    filtered_train_stats = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_examples=filtered_train,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=device,
        seed=args.seed,
    )
    filtered_val_acc, filtered_logs = evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        val_examples=val_examples,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        device=device,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "config": vars(args),
        "num_examples_total": len(all_examples),
        "num_examples_train": len(train_examples),
        "num_examples_val": len(val_examples),
        "curve_results": curves,
        "filtered_experiment": {
            "filtered_train_size": len(filtered_train),
            "val_answer_accuracy": filtered_val_acc,
            "train_stats": filtered_train_stats,
            "val_summary": filtered_logs["summary"],
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run SFT scaling/filter experiments for MATH-style data.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/a5-alignment/MATH/sft.jsonl",
        help="Path to reasoning SFT dataset jsonl",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default="/home/seanlinux/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B",
    )
    parser.add_argument("--output-path", type=str, default="sft_experiment_results.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_experiment(args)
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved results to {output_path}")
    print("Curve summary:")
    for item in result["curve_results"]:
        print(f"  train_size={item['train_size']}: val_answer_accuracy={item['val_answer_accuracy']:.4f}")
    print(
        "Filtered: "
        f"train_size={result['filtered_experiment']['filtered_train_size']}, "
        f"val_answer_accuracy={result['filtered_experiment']['val_answer_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()


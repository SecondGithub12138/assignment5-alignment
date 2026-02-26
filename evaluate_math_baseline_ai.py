"""
Evaluate zero-shot performance of Qwen 2.5 Math 1.5B on GSM8K dataset.

This script:
1. Loads GSM8K test examples from data/gsm8k/test.jsonl
2. Formats prompts using r1_zero prompt template (or question_only)
3. Generates outputs from the language model
4. Calculates evaluation metrics using r1_zero_reward_fn (or question_only_reward_fn)
5. Serializes results to disk for later analysis
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

# Import reward functions
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
# (2) format them as string prompts to the language model using the r1_zero prompt, and 
# (3) generate outputs for each example. This script should also 
# (4) calculate evaluation metrics and
# (5) serialize the examples, model generations, and corresponding evaluation scores to disk for
# analysis in subsequent problems.

# Load prompt template from file
R1_ZERO_PROMPT_PATH = Path(__file__).parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
QUESTION_ONLY_PROMPT_PATH = Path(__file__).parent / "cs336_alignment" / "prompts" / "question_only.prompt"

with open(R1_ZERO_PROMPT_PATH, "r") as f:
    r1_zero_prompt_template = f.read().strip()

with open(QUESTION_ONLY_PROMPT_PATH, "r") as f:
    question_only_prompt_template = f.read().strip()

logger = logging.getLogger(__name__)


def load_gsm8k_data(data_path: str) -> List[dict]:
    """
    Load GSM8K test examples from JSONL file.
    
    Args:
        data_path: Path to GSM8K test.jsonl file
        
    Returns:
        List of examples, each containing 'question' and 'answer' fields
    """
    examples = []
    with xopen(data_path) as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    logger.info(f"Loaded {len(examples)} examples from {data_path}")
    return examples


def format_r1_zero_prompt(question: str) -> str:
    """
    Format a question using the r1_zero prompt template.
    
    Args:
        question: The math problem question
        
    Returns:
        Formatted prompt string
    """
    return r1_zero_prompt_template.format(question=question)


def format_question_only_prompt(question: str) -> str:
    """
    Format a question using the question_only prompt template.
    
    Args:
        question: The math problem question
        
    Returns:
        Formatted prompt string
    """
    return question_only_prompt_template.format(question=question)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    examples: List[dict],
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    
    Args:
        vllm_model: Initialized vLLM LLM model
        reward_fn: Function that takes (response, ground_truth) and returns metrics dict
        prompts: List of formatted prompt strings
        eval_sampling_params: SamplingParams for generation
        ground_truths: List of ground truth answers
        examples: Original examples from dataset
        output_path: Path to write output JSONL file
    """
    logger.info(f"Generating outputs for {len(prompts)} prompts...")
    
    # Generate outputs using vLLM
    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Extract generated text from outputs
    generated_texts = []
    for output in raw_outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
    
    assert len(generated_texts) == len(prompts), "Mismatch between prompts and outputs"
    logger.info(f"Generated {len(generated_texts)} outputs")
    
    # Evaluate each output and serialize results
    all_metrics = []
    with xopen(output_path, "w") as fout:
        for example, prompt, generated_text, ground_truth in tqdm(
            zip(examples, prompts, generated_texts, ground_truths),
            total=len(examples),
            desc="Evaluating outputs"
        ):
            # Compute metrics using reward function
            # r1_zero_reward_fn signature: (response, ground_truth, fast=True)
            metrics = reward_fn(generated_text, ground_truth, fast=True)
            all_metrics.append(metrics)
            
            # Serialize result with example, prompt, generation, and metrics
            result = {
                **example,  # Include original example fields (problem, solution, etc.)
                "prompt": prompt,
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "metrics": metrics,
            }
            
            fout.write(json.dumps(result) + "\n")
    
    # Log summary statistics
    if all_metrics:
        format_rewards = [m["format_reward"] for m in all_metrics]
        answer_rewards = [m["answer_reward"] for m in all_metrics]
        overall_rewards = [m["reward"] for m in all_metrics]
        
        logger.info(f"Format reward: {sum(format_rewards) / len(format_rewards):.4f}")
        logger.info(f"Answer reward: {sum(answer_rewards) / len(answer_rewards):.4f}")
        logger.info(f"Overall reward: {sum(overall_rewards) / len(overall_rewards):.4f}")
        
        # Count categories
        correct_count = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 1.0)
        format_correct_answer_wrong = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 0.0)
        format_wrong_answer_wrong = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 0.0)
        format_wrong = sum(1 for m in all_metrics if m["format_reward"] == 0.0)
        
        logger.info(f"Correct (format=1, answer=1): {correct_count}")
        logger.info(f"Format correct, answer wrong (format=1, answer=0): {format_correct_answer_wrong}")
        logger.info(f"Format wrong, answer wrong (format=0, answer=0): {format_wrong_answer_wrong}")
        logger.info(f"Format wrong (format=0): {format_wrong}")
        
def extract_gsm8k_ground_truth(example: dict) -> str:
    """
    Extract ground truth answer from GSM8K example.
    
    GSM8K examples have an 'answer' field with format:
    "solution steps\n#### final_answer"
    
    We extract the final_answer after "#### "
    """
    answer_field = example.get("answer", "")
    
    # GSM8K format: "solution\n#### number"
    if "####" in answer_field:
        # Extract the number after "#### "
        parts = answer_field.split("####")
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            return final_answer
    
    # Fallback: return the whole answer field
    return answer_field.strip()


def main():
    """Main function to run the evaluation."""
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # Configuration
    gsm8k_test_path = str(Path(__file__).parent / "data" / "gsm8k" / "test.jsonl")
    model_path = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    output_path = "gsm8k_baseline_results_ai.jsonl"
    
    # Choose which prompt format and reward function to use
    use_r1_zero_format = True  # Set to False to use question_only format
    
    # Load GSM8K test data
    logger.info(f"Loading GSM8K test data from {gsm8k_test_path}...")
    examples = load_gsm8k_data(gsm8k_test_path)
    
    # Extract questions and ground truths
    questions = []
    ground_truths = []
    for example in examples:
        # GSM8K examples have 'question' field
        question = example.get("question", "")
        questions.append(question)
        
        # Extract ground truth answer
        ground_truth = extract_gsm8k_ground_truth(example)
        ground_truths.append(ground_truth)
    
    # Format prompts using chosen template
    logger.info(f"Formatting prompts using {'r1_zero' if use_r1_zero_format else 'question_only'} template...")
    if use_r1_zero_format:
        prompts = [format_r1_zero_prompt(question) for question in questions]
        reward_fn = r1_zero_reward_fn
    else:
        prompts = [format_question_only_prompt(question) for question in questions]
        reward_fn = question_only_reward_fn
    
    # Initialize vLLM model
    logger.info(f"Initializing vLLM model from {model_path}...")
    vllm_model = LLM(
        model=model_path,
    )
    
    # Set up sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=1.0,  # Deterministic for evaluation
        top_p=1.0,
        max_tokens=1024,  # Sufficient for math problems
        # stop=["\n"] if use_r1_zero_format else None,  # Stop at answer tag for r1_zero format
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=ground_truths,
        examples=examples,
        output_path=output_path,
    )
    
    logger.info(f"Evaluation complete! Results saved to {output_path}")
    
    # Clean up resources
    del vllm_model
    import gc
    gc.collect()


if __name__ == "__main__":
    main()

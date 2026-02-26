# Assignment 5: Qwen 2.5 Math 1.5B Evaluation Answers

## Part (b): Model Evaluation and Reward Function Analysis

### 1. Categorization of Model Generations

We evaluated the Qwen 2.5 Math 1.5B model on the GSM8K training set (7,473 examples) using the `r1_zero_reward_fn` function. The model generations were categorized into three groups:

**(1) Correct with both format and answer reward = 1:**
- **Count: 376 (5.03%)**
- **Description:** The model both followed the required format (containing `</think> <answer>` and `</answer>` tags) and provided the correct answer.

**(2) Format reward = 1 and answer reward = 0:**
- **Count: 1,595 (21.34%)**
- **Description:** The model correctly followed the format requirements but provided an incorrect answer.

**(3) Format reward = 0 and answer reward = 0:**
- **Count: 5,502 (73.63%)**
- **Description:** The model failed to follow the required format, and consequently the answer could not be properly evaluated or was incorrect.

### 2. Format Reward = 0 Analysis

**Observation of at least 10 cases:**

After examining multiple cases where `format_reward = 0`, we identified the following patterns:

**Pattern 1: Missing both format tags (most common, ~88%)**
- The model generates reasoning text but completely omits both `</think> <answer>` and `</answer>` tags.
- Example: Model generates "We need to calculate... Answer: 72" without any XML-style tags.

**Pattern 2: Missing closing answer tag (~10%)**
- The model generates `</think> <answer>` but fails to close with `</answer>`.
- Example: Model generates "reasoning text </think> <answer> 18 dollars" but stops before `</answer>`.

**Pattern 3: Incorrect tag format (~2%)**
- The model generates tags but in incorrect format or order.
- Example: Model generates `</think>` but not followed by `<answer>`.

**Root Cause Analysis:**

**This is primarily a MODEL OUTPUT problem, not a parser problem.**

**Why?**

1. **Zero-shot limitation:** The model has not been fine-tuned to generate the specific XML-style format required. While the prompt template clearly instructs the format (`<think> ... </think> <answer> ... </answer>`), the model in zero-shot mode struggles to consistently follow this instruction.

2. **Evidence from successful cases:** When the model does generate the correct format (26.37% of cases), it demonstrates that:
   - The parser correctly identifies the format when it exists
   - The model has the capability to understand and generate the format, but lacks consistency

3. **Parser validation:** The parser logic in `r1_zero_reward_fn` is straightforward:
   ```python
   if "</think> <answer>" in response and "</answer>" in response:
       # Format is correct
   ```
   This is a simple string matching check that works correctly when the format is present.

4. **Generation behavior:** Many format_wrong samples show the model generating natural language reasoning without any structured tags, suggesting the model defaults to free-form text generation rather than structured XML output.

**Conclusion:** The issue lies with the **base model's zero-shot output**. The model needs supervised fine-tuning (SFT) to learn to consistently generate the required format. The parser is functioning correctly and would work if the model generated the proper format.

### 3. Format Reward = 1, Answer Reward = 0 Analysis

**Observation of at least 10 cases:**

After examining cases where `format_reward = 1` but `answer_reward = 0`, we identified the following error types:

**Error Type 1: Unit/Format Issues (~30%)**
- The model includes units in the answer (e.g., "18 dollars", "3 quarts") while the ground truth is just the number (e.g., "18", "3").
- Example: Model answer: "18 dollars", Ground truth: "18"
- **Analysis:** This could be either:
  - A parser issue: The grading function may not properly normalize answers with units
  - A model issue: The model over-generates units when they're not needed
- **Verdict:** Likely a **parser limitation** - the grading function should handle unit normalization better, but the model also shouldn't add unnecessary units.

**Error Type 2: Calculation Errors (~60%)**
- The model understands the problem and follows the format correctly, but makes computational mistakes.
- Example: Model calculates "180" when the correct answer is "540", or uses incorrect mathematical operations.
- **Verdict:** This is a **model capability problem** - the model lacks sufficient mathematical reasoning ability to solve the problems correctly.

**Error Type 3: Reasoning Logic Errors (~10%)**
- The model misinterprets the problem or uses incorrect reasoning steps.
- Example: Model answers a different question or applies wrong problem-solving strategy.
- **Verdict:** This is a **model understanding problem** - the model fails to correctly understand or reason about the problem.

**Conclusion:**

- **Format correctness** demonstrates that the model can understand and follow format instructions when it does so (26.37% of cases), showing the parser works correctly when the format is present.
- **Answer errors** are primarily due to **model's limited mathematical reasoning capability**.
- Some cases (unit issues) may involve both parser limitations and model over-generation.
- Overall, this indicates the model needs:
  1. Better mathematical reasoning training (possibly through RL or more data)
  2. Improved answer normalization in the parser to handle units
  3. Better instruction following to avoid adding unnecessary units

### Summary Statistics

- **Average format_reward: 0.2637** (26.37% of samples have correct format)
- **Average answer_reward: 0.0503** (5.03% of samples have correct answers)
- **Average overall_reward: 0.0503** (5.03% of samples are fully correct)

**Key Insight:** Among the 26.37% of samples with correct format, only 19.1% (376 out of 1,971) have correct answers, indicating that even when the model follows the format, its mathematical reasoning accuracy is limited.

---

## Part (c): Zero-shot Baseline Performance

**How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on GSM8K?**

The Qwen 2.5 Math 1.5B model achieves a **5.03% overall accuracy** (format_reward=1 and answer_reward=1) on the GSM8K training set in zero-shot mode, with **26.37% format compliance** and **5.03% answer accuracy**. The model's performance is limited by two main factors: (1) its inability to consistently follow the required XML-style output format (73.63% format errors), and (2) its limited mathematical reasoning capability, as evidenced by only 19.1% accuracy among format-correct samples. These results indicate that the model requires supervised fine-tuning to learn the output format and further training to improve mathematical reasoning.

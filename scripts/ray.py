import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

from tests.data_util import sample_batch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.adapters import (
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_get_response_log_probs_chunked,
)

MODEL_NAME = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
WORKSPACE = "/home/seanlinux/assignment5-alignment"

GROUP_SIZE = 8
GRAD_ACCUM = 128
ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256
MICRO_BATCH = TRAIN_BATCH_SIZE // GRAD_ACCUM  # 2
DATA_SET_SIZE = ROLLOUT_BATCH_SIZE // GROUP_SIZE  # 32
SAMPLING_MAX_TOKENS = 1024
SAMPLING_MIN_TOKENS = 4
N_GRPO_STEPS = 1000
LR = 3e-5
ADVANTAGE_EPS = 1e-6
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        import sys
        sys.path.insert(0, WORKSPACE)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.device = torch.device("cuda:0")

        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for weight transfer."""
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )

    def train_step(self, repeated_prompt_strs, repeated_ground_truths,
                   generated_texts, grpo_step):
        """Full GRPO training step: rewards → old_log_probs → grad accum → optimizer.step."""
        self.model.gradient_checkpointing_enable()
        self.tokenizer.padding_side = "right"

        advantages, raw_rewards, _ = run_compute_group_normalized_rewards(
            r1_zero_reward_fn, generated_texts, repeated_ground_truths,
            GROUP_SIZE, ADVANTAGE_EPS, USE_STD_NORMALIZATION,
        )

        all_train_data = run_tokenize_prompt_and_output(
            repeated_prompt_strs, generated_texts, self.tokenizer,
        )

        with torch.inference_mode():
            all_old_log_probs = run_get_response_log_probs_chunked(
                self.model, all_train_data["input_ids"], all_train_data["labels"],
                return_token_entropy=False, chunk_size=8, device=self.device,
            )["log_probs"].to(self.device)

        train_start = time.time()
        for grad_step in range(GRAD_ACCUM):
            s = grad_step * MICRO_BATCH
            e = s + MICRO_BATCH

            input_ids = all_train_data["input_ids"][s:e].to(self.device)
            labels = all_train_data["labels"][s:e].to(self.device)
            response_mask = all_train_data["response_mask"][s:e].to(self.device)
            micro_old_lp = all_old_log_probs[s:e]
            micro_adv = advantages[s:e].unsqueeze(-1).to(self.device)
            micro_rew = raw_rewards[s:e].unsqueeze(-1).to(self.device)

            log_probs = run_get_response_log_probs(
                self.model, input_ids, labels, False,
            )["log_probs"]

            loss, _ = run_grpo_microbatch_train_step(
                log_probs, response_mask, GRAD_ACCUM, LOSS_TYPE,
                micro_rew, micro_adv, micro_old_lp, 1,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.gradient_checkpointing_disable()

        train_time = time.time() - train_start
        reward_mean = raw_rewards.float().mean().item()
        print(f"[grpo_step {grpo_step}/{N_GRPO_STEPS}] "
              f"loss={float(loss):.6f} reward={reward_mean:.4f} train={train_time:.1f}s")
        return {"loss": float(loss), "reward_mean": reward_mean, "train_time": train_time}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def prepare_rollout_data():
    """Sample new prompts, repeat for group_size, extract ground truths."""
    prompt_strs, output_strs = sample_batch(DATA_SET_SIZE)
    repeated_prompts = [p for p in prompt_strs for _ in range(GROUP_SIZE)]
    ground_truths = [
        o.split("<answer>")[-1].replace("</answer>", "").strip()
        for o in output_strs
    ]
    repeated_gts = [t for t in ground_truths for _ in range(GROUP_SIZE)]
    return repeated_prompts, repeated_gts


def extract_texts(rollout_results):
    """Extract generated text strings from vLLM RequestOutput objects."""
    return [r.outputs[0].text for r in rollout_results]


def transfer_weights():
    """Sync training model weights → vLLM inference engine via NCCL."""
    names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())
    inference_handle = llm.update_weights.remote(
        dict(update_info=dict(
            names=names, dtype_names=dtype_names,
            shapes=shapes, packed=True,
        ))
    )
    train_handle = train_model.broadcast_weights.remote(packed=True)
    ray.get([train_handle, inference_handle])


# ===========================================================================
# Main
# ===========================================================================
ray.init()

# 1) Launch training actor (1 GPU)
train_model = TrainModel.remote(MODEL_NAME)

# 2) Reserve 2 GPUs for vLLM inference via placement group
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# 3) Launch vLLM with dummy weights (will sync real weights before first rollout)
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    dtype=torch.bfloat16,
    tensor_parallel_size=2,
    data_parallel_size=1,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
)

sampling_params = SamplingParams(
    temperature=1.0,
    min_tokens=SAMPLING_MIN_TOKENS,
    max_tokens=SAMPLING_MAX_TOKENS,
)

# 4) Setup NCCL weight transfer channel
master_address, master_port = ray.get(
    train_model.get_master_address_and_port.remote()
)
world_size = ray.get(llm.get_world_size.remote()) + 1  # +1 for trainer
inference_handle = llm.init_weight_transfer_engine.remote(
    dict(init_info=dict(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,
        world_size=world_size,
    ))
)
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])

# 5) Initial weight sync (dummy → real weights) + first rollout
transfer_weights()
prompts, ground_truths = prepare_rollout_data()
rollout_results = ray.get(llm.generate.remote(prompts, sampling_params))
generated_texts = extract_texts(rollout_results)

# 6) GRPO training loop — train + next rollout overlap
#
#    Timeline per step:
#      ┌─ train on current data (train GPU) ─┐
#      │                                      │
#      ├─ generate next rollout (vLLM GPUs) ──┤
#      └──────────────────────────────────────┘
#      │ transfer_weights (both sides block)  │
#
#    The next rollout uses weights from BEFORE this step's training
#    (1-step off-policy), which GRPO clip loss handles correctly.

for step in range(N_GRPO_STEPS):
    # Async: train on current rollout data
    train_handle = train_model.train_step.remote(
        prompts, ground_truths, generated_texts, step,
    )

    # Async: sample new data + start next rollout (uses pre-training weights)
    next_prompts, next_ground_truths = prepare_rollout_data()
    next_rollout_handle = llm.generate.remote(next_prompts, sampling_params)

    # Wait for both to finish
    metrics, next_rollout_results = ray.get([train_handle, next_rollout_handle])
    print(f"[step {step}] {metrics}")

    # Sync trained weights to vLLM
    transfer_weights()

    # Advance to next iteration
    prompts = next_prompts
    ground_truths = next_ground_truths
    generated_texts = extract_texts(next_rollout_results)

print("Training complete!")

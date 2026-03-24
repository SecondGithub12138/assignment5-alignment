import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.adapters import (
    run_compute_group_normalized_rewards,
    run_get_response_log_probs,
    run_get_response_log_probs_chunked,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
)
from tests.data_util import sample_batch

MODEL_NAME = "/home/seanlinux/assignment5-alignment/models/Qwen2.5-Math-1.5B"
WORKSPACE = "/home/seanlinux/assignment5-alignment"

TRAIN_GPU_COUNT = 32
VLLM_TP_SIZE = 2
VLLM_DP_SIZE = 8
VLLM_GPU_COUNT = VLLM_TP_SIZE * VLLM_DP_SIZE

GROUP_SIZE = 8
ROLLOUT_BATCH_SIZE = 256
MICRO_BATCH = 2
DATA_SET_SIZE = ROLLOUT_BATCH_SIZE // GROUP_SIZE
SAMPLING_MAX_TOKENS = 1024
SAMPLING_MIN_TOKENS = 4
N_GRPO_STEPS = 1000
LR = 3e-5
ADVANTAGE_EPS = 1e-6
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True


def _node_resource_key(resources: dict[str, float]) -> str:
    for key in resources:
        if key.startswith("node:"):
            return key
    raise RuntimeError(f"Missing node:* resource key in {resources}")


def discover_gpu_nodes() -> list[dict[str, Any]]:
    """
    Input:
      - none (reads `ray.nodes()` internally)

    Output:
      - list[dict], each dict has:
        {
          "ip": str,
          "gpus": int,
          "resource_key": str,   # e.g. "node:10.0.0.12"
        }

    Example output:
      [
        {"ip": "10.0.0.1", "gpus": 8, "resource_key": "node:10.0.0.1"},
        {"ip": "10.0.0.2", "gpus": 8, "resource_key": "node:10.0.0.2"},
        {"ip": "10.0.0.5", "gpus": 4, "resource_key": "node:10.0.0.5"},
      ]
    """
    nodes: list[dict[str, Any]] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        gpu_count = int(resources.get("GPU", 0))
        if gpu_count <= 0:
            continue
        nodes.append(
            {
                "ip": node.get("NodeManagerAddress", "unknown"),
                "gpus": gpu_count,
                "resource_key": _node_resource_key(resources),
            }
        )
    nodes.sort(key=lambda x: (-x["gpus"], x["ip"]))
    return nodes


def _allocate_gpus_on_disjoint_nodes(
    nodes: list[dict[str, Any]], required_gpus: int
) -> tuple[list[tuple[dict[str, Any], int]], list[dict[str, Any]]]:
    """
    Greedy allocator on sorted nodes.
    Returns (allocation, remaining_nodes).
    """
    if required_gpus <= 0:
        return [], nodes

    remaining = required_gpus
    allocation: list[tuple[dict[str, Any], int]] = []
    used_keys = set()
    for node in nodes:
        if remaining == 0:
            break
        take = min(node["gpus"], remaining)
        if take <= 0:
            continue
        allocation.append((node, take))
        used_keys.add(node["resource_key"])
        remaining -= take

    if remaining > 0:
        total = sum(n["gpus"] for n in nodes)
        raise RuntimeError(
            f"Not enough GPUs: need {required_gpus}, have {total} across nodes={nodes}"
        )

    remaining_nodes = [n for n in nodes if n["resource_key"] not in used_keys]
    return allocation, remaining_nodes


def choose_layout(
    nodes: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], int]], list[tuple[dict[str, Any], int]]]:
    """
    Input:
      - nodes: list[dict], usually the output of `discover_gpu_nodes()`

    Output:
      - (train_alloc, rollout_alloc)
      - each is list[(node_dict, gpu_count_to_take)]

    Example output shape:
      train_alloc = [
        ({"ip": "10.0.0.1", "gpus": 8, "resource_key": "node:10.0.0.1"}, 8),
        ({"ip": "10.0.0.2", "gpus": 8, "resource_key": "node:10.0.0.2"}, 8),
      ]
      rollout_alloc = [
        ({"ip": "10.0.0.3", "gpus": 8, "resource_key": "node:10.0.0.3"}, 4),
      ]

    Generic disjoint layout:
      - train gets TRAIN_GPU_COUNT GPUs first
      - rollout gets VLLM_GPU_COUNT GPUs from remaining nodes
    """
    train_alloc, remaining_nodes = _allocate_gpus_on_disjoint_nodes(
        nodes, TRAIN_GPU_COUNT
    )
    rollout_alloc, _ = _allocate_gpus_on_disjoint_nodes(
        remaining_nodes, VLLM_GPU_COUNT
    )
    return train_alloc, rollout_alloc


def build_node_pinned_bundles(node_alloc: list[tuple[dict[str, Any], int]]) -> list[dict[str, float]]:
    """
    Convert node allocation plan into Ray placement-group bundles.

    Example input:
      node_alloc = [
        ({"ip": "10.0.0.1", "resource_key": "node:10.0.0.1", "gpus": 8}, 2),
        ({"ip": "10.0.0.2", "resource_key": "node:10.0.0.2", "gpus": 4}, 1),
      ]

    Example output:
      [
        {"GPU": 1.0, "CPU": 0.0, "node:10.0.0.1": 0.001},
        {"GPU": 1.0, "CPU": 0.0, "node:10.0.0.1": 0.001},
        {"GPU": 1.0, "CPU": 0.0, "node:10.0.0.2": 0.001},
      ]

    Meaning:
    - one bundle reserves one GPU
    - node:<ip> key pins that bundle to the target machine
    """
    bundles: list[dict[str, float]] = []
    for node, count in node_alloc:
        for _ in range(count):
            bundles.append(
                {
                    "GPU": 1.0,
                    "CPU": 0.0,
                    node["resource_key"]: 0.001,
                }
            )
    return bundles


@ray.remote
class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            str(i) for i in range(VLLM_GPU_COUNT)
        )
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, num_cpus=0)
class TrainWorker:
    def __init__(
        self,
        model_name: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        sys.path.insert(0, WORKSPACE)

        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        auto_wrap = transformer_auto_wrap_policy(
            transformer_layer_cls={Qwen2DecoderLayer}
        )
        self.model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap,
            device_id=self.device,
            use_orig_params=True,
            sync_module_states=True,
        )
        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        self.optimizer.zero_grad(set_to_none=True)

        self.transfer_master_address = get_ip()
        self.transfer_port = get_open_port()
        self.model_update_group = None
        self._cached_weight_meta = self._build_weight_metadata()

    def _build_weight_metadata(self):
        cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            full_state = self.model.state_dict()

        if self.rank != 0:
            return [], [], []

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, tensor in full_state.items():
            if not torch.is_tensor(tensor):
                continue
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))
        return names, dtype_names, shapes

    def get_transfer_master_address_and_port(self):
        if self.rank != 0:
            raise RuntimeError("Only rank0 serves transfer address/port")
        return self.transfer_master_address, self.transfer_port

    def init_weight_transfer_group_rank0(self, world_size: int):
        if self.rank == 0:
            self.model_update_group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=self.transfer_master_address,
                    master_port=self.transfer_port,
                    world_size=world_size,
                )
            )
        return True

    def get_weight_metadata(self):
        if self.rank != 0:
            raise RuntimeError("Only rank0 serves weight metadata")
        return self._cached_weight_meta

    def broadcast_weights_rank0(self, packed: bool = True):
        cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            full_state = self.model.state_dict()

        if self.rank == 0:
            if self.model_update_group is None:
                raise RuntimeError("Rank0 transfer group is not initialized")
            names, _, _ = self._cached_weight_meta
            iterator = (
                (name, full_state[name])
                for name in names
                if name in full_state and torch.is_tensor(full_state[name])
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iterator,
                group=self.model_update_group,
                packed=packed,
            )
        dist.barrier()
        return True

    def _slice_rank_equal(self, tensor: torch.Tensor, local_batch: int):
        start = self.rank * local_batch
        end = start + local_batch
        return tensor[start:end]

    def train_step(
        self,
        repeated_prompt_strs: list[str],
        repeated_ground_truths: list[str],
        generated_texts: list[str],
        grpo_step: int,
    ) -> dict[str, Any]:
        self.model.train()
        if hasattr(self.model, "module") and hasattr(self.model.module, "gradient_checkpointing_enable"):
            self.model.module.gradient_checkpointing_enable()
        self.tokenizer.padding_side = "right"

        advantages, raw_rewards, _ = run_compute_group_normalized_rewards(
            r1_zero_reward_fn,
            generated_texts,
            repeated_ground_truths,
            GROUP_SIZE,
            ADVANTAGE_EPS,
            USE_STD_NORMALIZATION,
        )
        all_train_data = run_tokenize_prompt_and_output(
            repeated_prompt_strs, generated_texts, self.tokenizer
        )

        global_batch = int(all_train_data["input_ids"].shape[0])
        local_batch = global_batch // self.world_size
        if local_batch == 0:
            raise RuntimeError(
                f"global batch {global_batch} is smaller than world_size {self.world_size}"
            )
        effective_batch = local_batch * self.world_size
        if self.rank == 0 and effective_batch != global_batch:
            print(
                f"[train_step {grpo_step}] dropping {global_batch - effective_batch} "
                "samples for equal per-rank batch size"
            )

        all_input_ids = all_train_data["input_ids"][:effective_batch]
        all_labels = all_train_data["labels"][:effective_batch]
        all_response_mask = all_train_data["response_mask"][:effective_batch]
        advantages = advantages[:effective_batch]
        raw_rewards = raw_rewards[:effective_batch]

        input_ids_local = self._slice_rank_equal(all_input_ids, local_batch).to(self.device)
        labels_local = self._slice_rank_equal(all_labels, local_batch).to(self.device)
        response_mask_local = self._slice_rank_equal(all_response_mask, local_batch).to(self.device)
        advantages_local = self._slice_rank_equal(advantages, local_batch).to(self.device)
        raw_rewards_local = self._slice_rank_equal(raw_rewards, local_batch).to(self.device)

        with torch.inference_mode():
            old_log_probs_local = run_get_response_log_probs_chunked(
                self.model,
                input_ids_local,
                labels_local,
                return_token_entropy=False,
                chunk_size=8,
                device=self.device,
            )["log_probs"].to(self.device)

        local_grad_accum = (local_batch + MICRO_BATCH - 1) // MICRO_BATCH
        train_start = time.time()
        for grad_step in range(local_grad_accum):
            s = grad_step * MICRO_BATCH
            e = min(s + MICRO_BATCH, local_batch)
            if s >= e:
                continue

            input_ids = input_ids_local[s:e]
            labels = labels_local[s:e]
            response_mask = response_mask_local[s:e]
            micro_old_lp = old_log_probs_local[s:e]
            micro_adv = advantages_local[s:e].unsqueeze(-1)
            micro_rew = raw_rewards_local[s:e].unsqueeze(-1)

            log_probs = run_get_response_log_probs(
                self.model,
                input_ids,
                labels,
                False,
            )["log_probs"]

            run_grpo_microbatch_train_step(
                log_probs,
                response_mask,
                local_grad_accum,
                LOSS_TYPE,
                micro_rew,
                micro_adv,
                micro_old_lp,
                1,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        dist.barrier()

        train_time = time.time() - train_start
        reward_mean = raw_rewards.float().mean().item()
        if self.rank == 0:
            print(
                f"[grpo_step {grpo_step}/{N_GRPO_STEPS}] "
                f"reward={reward_mean:.4f} train={train_time:.1f}s "
                f"(local_batch={local_batch})"
            )
            return {"reward_mean": reward_mean, "train_time": train_time}
        return {"rank": self.rank, "reward_mean": reward_mean, "train_time": train_time}

    def shutdown(self):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return True


def prepare_rollout_data():
    prompt_strs, output_strs = sample_batch(DATA_SET_SIZE)
    repeated_prompts = [p for p in prompt_strs for _ in range(GROUP_SIZE)]
    ground_truths = [
        o.split("<answer>")[-1].replace("</answer>", "").strip()
        for o in output_strs
    ]
    repeated_gts = [t for t in ground_truths for _ in range(GROUP_SIZE)]
    return repeated_prompts, repeated_gts


def extract_texts(rollout_results):
    return [r.outputs[0].text for r in rollout_results]


def transfer_weights(train_workers, llm):
    names, dtype_names, shapes = ray.get(train_workers[0].get_weight_metadata.remote())
    inference_handle = llm.update_weights.remote(
        dict(
            update_info=dict(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=True,
            )
        )
    )
    train_handles = [w.broadcast_weights_rank0.remote(packed=True) for w in train_workers]
    ray.get(train_handles + [inference_handle])


def connect_ray():
    try:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)
    except Exception:
        ray.init(ignore_reinit_error=True)


def main():
    connect_ray()

    gpu_nodes = discover_gpu_nodes()
    train_alloc, rollout_alloc = choose_layout(gpu_nodes)
    print(f"[nodes] discovered={gpu_nodes}")
    print(f"[nodes] train_alloc={[(n['ip'], c) for n, c in train_alloc]}")
    print(f"[nodes] rollout_alloc={[(n['ip'], c) for n, c in rollout_alloc]}")

    train_bundles = build_node_pinned_bundles(train_alloc)
    pg_train = placement_group(train_bundles)
    ray.get(pg_train.ready())

    train_master_addr = get_ip()
    train_master_port = get_open_port()
    train_workers = []
    for rank in range(TRAIN_GPU_COUNT):
        scheduling_train = PlacementGroupSchedulingStrategy(
            placement_group=pg_train,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=rank,
        )
        worker = TrainWorker.options(scheduling_strategy=scheduling_train).remote(
            MODEL_NAME,
            rank,
            TRAIN_GPU_COUNT,
            train_master_addr,
            train_master_port,
        )
        train_workers.append(worker)
    print(f"[resource layout] train={TRAIN_GPU_COUNT} GPUs (fsdp world_size=32)")

    rollout_bundles = build_node_pinned_bundles(rollout_alloc)
    pg_inference = placement_group(rollout_bundles)
    ray.get(pg_inference.ready())

    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    llm = MyLLM.options(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    ).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=VLLM_TP_SIZE,
        data_parallel_size=VLLM_DP_SIZE,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
        load_format="dummy",
    )
    print(f"[resource layout] vllm={VLLM_GPU_COUNT} GPUs (tp={VLLM_TP_SIZE}, dp={VLLM_DP_SIZE})")

    sampling_params = SamplingParams(
        temperature=1.0,
        min_tokens=SAMPLING_MIN_TOKENS,
        max_tokens=SAMPLING_MAX_TOKENS,
    )

    transfer_master_addr, transfer_master_port = ray.get(
        train_workers[0].get_transfer_master_address_and_port.remote()
    )
    world_size = ray.get(llm.get_world_size.remote()) + 1
    inference_handle = llm.init_weight_transfer_engine.remote(
        dict(
            init_info=dict(
                master_address=transfer_master_addr,
                master_port=transfer_master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )
    )
    ray.get(inference_handle)
    ray.get(train_workers[0].init_weight_transfer_group_rank0.remote(world_size))

    transfer_weights(train_workers, llm)
    prompts, ground_truths = prepare_rollout_data()
    rollout_results = ray.get(llm.generate.remote(prompts, sampling_params))
    generated_texts = extract_texts(rollout_results)

    for step in range(N_GRPO_STEPS):
        train_handles = [
            w.train_step.remote(prompts, ground_truths, generated_texts, step)
            for w in train_workers
        ]
        next_prompts, next_ground_truths = prepare_rollout_data()
        next_rollout_handle = llm.generate.remote(next_prompts, sampling_params)

        train_metrics_all = ray.get(train_handles)
        next_rollout_results = ray.get(next_rollout_handle)
        print(f"[step {step}] rank0_metrics={train_metrics_all[0]}")

        transfer_weights(train_workers, llm)
        prompts = next_prompts
        ground_truths = next_ground_truths
        generated_texts = extract_texts(next_rollout_results)

    ray.get([w.shutdown.remote() for w in train_workers])
    print("Training complete!")


if __name__ == "__main__":
    main()

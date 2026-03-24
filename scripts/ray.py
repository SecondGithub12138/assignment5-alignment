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

TRAIN_GPU_COUNT = 20
VLLM_TP_SIZE = 2
VLLM_DP_SIZE = 6
VLLM_GPU_COUNT = VLLM_TP_SIZE * VLLM_DP_SIZE

GROUP_SIZE = 8
ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256
MICRO_BATCH = 2
DATA_SET_SIZE = ROLLOUT_BATCH_SIZE // GROUP_SIZE
SAMPLING_MAX_TOKENS = 1024
SAMPLING_MIN_TOKENS = 4
N_GRPO_STEPS = 1000
LR = 3e-5
ADVANTAGE_EPS = 1e-6
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True


class MyLLM(LLM):
    """Configure vLLM worker placement under Ray."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            str(i) for i in range(VLLM_GPU_COUNT)
        )
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, num_cpus=0)
class TrainWorker:
    """One rank in the 20-GPU train-side FSDP process group."""

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

        names, dtype_names, shapes = [], [], []
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
        """
        All ranks participate in FSDP full-state gather.
        Rank0 streams gathered weights to vLLM.
        """
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

    def _slice_rank_equal(self, tensor: torch.Tensor, local_batch: int) -> torch.Tensor:
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
                f"[train_step {grpo_step}] dropping {global_batch - effective_batch} samples "
                "to keep per-rank batch size equal"
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


def main():
    ray.init()

    pg_train = placement_group([{"GPU": 1, "CPU": 0}] * TRAIN_GPU_COUNT)
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
    print(f"[resource layout] train={TRAIN_GPU_COUNT} GPUs (fsdp world_size=20)")

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * VLLM_GPU_COUNT)
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyLLM).remote(
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
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
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

TRAIN_GPU_COUNT = 20
VLLM_TP_SIZE = 2
VLLM_DP_SIZE = 6
VLLM_GPU_COUNT = VLLM_TP_SIZE * VLLM_DP_SIZE

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
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            str(i) for i in range(VLLM_GPU_COUNT)
        )
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, num_cpus=0)
class TrainWorker:
    """One rank in the train-side 20-GPU FSDP process group."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        import sys
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
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            full_state = self.model.state_dict()
        if self.rank != 0:
            return [], [], []

        names, dtype_names, shapes = [], [], []
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
        """
        All ranks participate in full-state all-gather.
        Rank0 sends the gathered tensors to vLLM.
        """
        cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            full_state = self.model.state_dict()

        if self.rank == 0:
            if self.model_update_group is None:
                raise RuntimeError("Rank0 weight transfer group is not initialized")
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

    def train_step(
        self,
        repeated_prompt_strs: list[str],
        repeated_ground_truths: list[str],
        generated_texts: list[str],
        grpo_step: int,
    ) -> dict[str, Any]:
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
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

        with torch.inference_mode():
            all_old_log_probs = run_get_response_log_probs_chunked(
                self.model,
                all_train_data["input_ids"],
                all_train_data["labels"],
                return_token_entropy=False,
                chunk_size=8,
                device=self.device,
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
                self.model,
                input_ids,
                labels,
                False,
            )["log_probs"]

            run_grpo_microbatch_train_step(
                log_probs,
                response_mask,
                GRAD_ACCUM,
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
                f"reward={reward_mean:.4f} train={train_time:.1f}s"
            )
            return {"reward_mean": reward_mean, "train_time": train_time}
        return {"rank": self.rank, "reward_mean": reward_mean, "train_time": train_time}

    def shutdown(self):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return True


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


def main():
    ray.init()

    pg_train = placement_group([{"GPU": 1, "CPU": 0}] * TRAIN_GPU_COUNT)
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
        worker = TrainWorker.options(
            scheduling_strategy=scheduling_train
        ).remote(
            MODEL_NAME,
            rank,
            TRAIN_GPU_COUNT,
            train_master_addr,
            train_master_port,
        )
        train_workers.append(worker)
    print(f"[resource layout] train={TRAIN_GPU_COUNT} GPUs (fsdp world_size=20)")

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * VLLM_GPU_COUNT)
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyLLM).remote(
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
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import ray
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType
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
TRAIN_GPU_COUNT = 20
VLLM_TP_SIZE = 2
VLLM_DP_SIZE = 6
VLLM_GPU_COUNT = VLLM_TP_SIZE * VLLM_DP_SIZE

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
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            str(i) for i in range(VLLM_GPU_COUNT)
        )
        super().__init__(*args, **kwargs)


@ray.remote
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

    def broadcast_weights_rank0(self, packed: bool = True):
        """Broadcast rank-0 full weights to the inference engine."""
        if isinstance(self.model, FullyShardedDataParallel):
            # For FSDP, gather a full state dict on rank-0 before transfer.
            cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            with FullyShardedDataParallel.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                cfg,
            ):
                full_state = self.model.state_dict()
            iterator = (
                (name, tensor)
                for name, tensor in full_state.items()
                if torch.is_tensor(tensor)
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iterator,
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
    """Sync training rank-0 full weights → vLLM via NCCL."""
    names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())
    inference_handle = llm.update_weights.remote(
        dict(update_info=dict(
            names=names, dtype_names=dtype_names,
            shapes=shapes, packed=True,
        ))
    )
    train_handle = train_model.broadcast_weights_rank0.remote(packed=True)
    ray.get([train_handle, inference_handle])


# ===========================================================================
# Main
# ===========================================================================
ray.init()

# 1) Reserve 20 GPUs for training actor.
pg_train = placement_group([{"GPU": 1, "CPU": 0}] * TRAIN_GPU_COUNT)
ray.get(pg_train.ready())
scheduling_train = PlacementGroupSchedulingStrategy(
    placement_group=pg_train,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)
train_model = TrainModel.options(
    num_gpus=1,
    num_cpus=0,
    scheduling_strategy=scheduling_train,
).remote(MODEL_NAME)
print(
    f"[resource layout] train_reserved={TRAIN_GPU_COUNT} GPUs "
    "(target fsdp world_size=20; current trainer actor uses 1 GPU)"
)

# 2) Reserve 12 GPUs for vLLM inference via placement group
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * VLLM_GPU_COUNT)
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
    tensor_parallel_size=VLLM_TP_SIZE,
    data_parallel_size=VLLM_DP_SIZE,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
)
print(
    f"[resource layout] vllm={VLLM_GPU_COUNT} GPUs "
    f"(tp={VLLM_TP_SIZE}, dp={VLLM_DP_SIZE})"
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

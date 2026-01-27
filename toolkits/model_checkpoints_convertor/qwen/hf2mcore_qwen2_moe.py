# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DEBUG FLAG - Set to True for detailed memory tracking, False for production
ENABLE_DEBUG = False

import gc
import math
import sys
import os
import re
import torch
import warnings
from collections import defaultdict
import psutil
import threading
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2MoeForCausalLM,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from qwen2.pretrain_qwen2_moe import model_provider
from megatron_patch.arguments import get_patch_args

from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
)

from megatron.core.models.gpt import GPTModel


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

warnings.filterwarnings("ignore", category=UserWarning)

# Memory monitoring utilities
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb / 1024

def print_memory_checkpoint(location):
    """Print memory usage at a specific location"""
    memory_gb = get_memory_usage()
    print(f"🔍 MEMORY DEBUG [{location}]: {memory_gb:.2f} GB")
    return memory_gb

# Global memory tracking
_memory_history = []
def track_memory(location):
    """Track memory and store in history"""
    memory_gb = print_memory_checkpoint(location)
    _memory_history.append((location, memory_gb))
    return memory_gb

def get_model_memory_size(model):
    """Calculate memory size of a model in GB"""
    if not ENABLE_DEBUG or model is None:
        return 0.0

    total_size = 0
    param_count = 0

    # Try to get model parameters
    try:
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                if param is not None and hasattr(param, 'data'):
                    param_size = param.data.numel() * param.data.element_size()
                    total_size += param_size
                    param_count += 1
        elif hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            for name, tensor in state_dict.items():
                if tensor is not None:
                    tensor_size = tensor.numel() * tensor.element_size()
                    total_size += tensor_size
                    param_count += 1
    except Exception as e:
        if ENABLE_DEBUG:
            print(f"⚠️  Error calculating model size: {e}")
        return 0.0

    size_gb = total_size / (1024**3)
    return size_gb

def get_layer_memory_size(layer):
    """Calculate memory size of a single layer in MB"""
    if not ENABLE_DEBUG or layer is None:
        return 0.0

    total_size = 0
    try:
        for param in layer.parameters():
            if param is not None and hasattr(param, 'data'):
                param_size = param.data.numel() * param.data.element_size()
                total_size += param_size
    except Exception as e:
        if ENABLE_DEBUG:
            print(f"⚠️  Error calculating layer size: {e}")
        return 0.0

    size_mb = total_size / (1024**2)
    return size_mb

def track_memory_with_models(location, hf_model=None, mg_model=None):
    """Track memory usage along with model sizes"""
    if not ENABLE_DEBUG:
        return track_memory(location)

    memory_gb = get_memory_usage()

    # Calculate model sizes
    hf_size = get_model_memory_size(hf_model) if hf_model is not None else 0.0
    mg_size = get_model_memory_size(mg_model) if mg_model is not None else 0.0

    # Compact format showing System | HF | MG | Total
    hf_indicator = "meta✓" if hf_size == 0.0 else f"{hf_size:.2f}"
    models_total = hf_size + mg_size

    print(f"🔍 MEMORY [{location}]: System={memory_gb:.2f}GB | HF={hf_indicator} | MG={mg_size:.2f} | Total={models_total:.2f}")

    _memory_history.append((location, memory_gb, hf_size, mg_size))
    return memory_gb, hf_size, mg_size

def track_layer_memory(layer_idx, total_layers, hf_layer=None, mg_layer=None):
    """Track memory usage for individual layers"""
    if not ENABLE_DEBUG:
        return

    memory_gb = get_memory_usage()

    hf_layer_size = get_layer_memory_size(hf_layer) if hf_layer is not None else 0.0
    mg_layer_size = get_layer_memory_size(mg_layer) if mg_layer is not None else 0.0

    print(f"🔄 LAYER {layer_idx+1}/{total_layers} MEMORY:")
    print(f"  📊 System Memory: {memory_gb:.2f} GB")
    print(f"  🤗 HF Layer: {hf_layer_size:.1f} MB")
    print(f"  ⚡ MG Layer: {mg_layer_size:.1f} MB")

    return memory_gb, hf_layer_size, mg_layer_size

def print_memory_summary():
    """Print a complete summary of memory usage throughout the conversion"""
    if not ENABLE_DEBUG:
        return

    print("\n" + "="*80)
    print("🔍 COMPLETE MEMORY USAGE SUMMARY")
    print("="*80)

    if not _memory_history:
        print("No memory tracking data available.")
        return

    max_memory = 0
    min_memory = float('inf')

    for entry in _memory_history:
        if len(entry) == 2:  # Old format: (location, memory_gb)
            location, memory_gb = entry
            print(f"📊 {location:<40}: System={memory_gb:.2f}GB")
            max_memory = max(max_memory, memory_gb)
            min_memory = min(min_memory, memory_gb)
        elif len(entry) == 4:  # New format: (location, memory_gb, hf_size, mg_size)
            location, memory_gb, hf_size, mg_size = entry
            total_models = hf_size + mg_size
            # Highlight when HF storage is 0 (meta device success)
            hf_indicator = "meta✓" if hf_size == 0.0 else f"{hf_size:.2f}"
            print(f"📊 {location:<40}: System={memory_gb:.2f}GB | HF={hf_indicator} | MG={mg_size:.2f} | Total={total_models:.2f}")
            max_memory = max(max_memory, memory_gb)
            min_memory = min(min_memory, memory_gb)

    print("="*80)
    print(f"📈 Peak Memory Usage: {max_memory:.2f} GB")
    print(f"📉 Minimum Memory Usage: {min_memory:.2f} GB")
    print(f"📊 Memory Range: {max_memory - min_memory:.2f} GB")
    print("="*80)


def _get_module_and_leaf(model, dotted):
    parts = dotted.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    return mod, leaf


def _assign_param_by_name(model, dotted, tensor_cpu):
    """Replace (possibly meta) parameter with CPU tensor while preserving requires_grad."""
    mod, leaf = _get_module_and_leaf(model, dotted)
    old = mod._parameters.get(leaf, None)
    requires_grad = bool(old.requires_grad) if old is not None else False
    mod._parameters[leaf] = torch.nn.Parameter(tensor_cpu, requires_grad=requires_grad)


def _assign_buffer_by_name(model, dotted, tensor_cpu):
    """Register or replace buffer with provided CPU tensor."""
    mod, leaf = _get_module_and_leaf(model, dotted)
    mod.register_buffer(leaf, tensor_cpu)


def _to_cpu_dtype(tensor, dtype):
    """Return fresh CPU tensor with requested dtype (no shared storage)."""
    return tensor.to(dtype=dtype, device="cpu", copy=True)


def _free_storage_safe(tensor):
    try:
        if isinstance(tensor, torch.Tensor):
            tensor.untyped_storage().resize_(0)
    except Exception:
        pass


def _assert_no_meta(hf_model):
    for name, param in hf_model.named_parameters():
        assert param.device.type != "meta", f"Parameter still on meta: {name}"
    for name, buf in hf_model.named_buffers():
        assert buf.device.type != "meta", f"Buffer still on meta: {name}"


def generate_rank_group(
        tensor_model_parallel_size,
        expert_tensor_parallel_size,
        expert_model_parallel_size,
        pipeline_model_parallel_size
):
    """
        copy from toolkits/model_checkpoints_convertor/deepseek/hf2mcore_deepseek_v3_moe.py

        This function attempts to generate group rank on the minimal practicable world_size.
        Support Decoder-Only model currently.
    """
    tp, etp, ep, pp = (
        tensor_model_parallel_size,
        expert_tensor_parallel_size,
        expert_model_parallel_size,
        pipeline_model_parallel_size
    )
    minimal_worldsize = pp * math.lcm(tp, etp * ep)
    print(f"The given parallel config should be run on at least {minimal_worldsize} cards")
    dp = minimal_worldsize // (pp * tp)
    edp = minimal_worldsize // (pp * ep * etp)
    # NOTE: If user want to scale up cp_size, he should downscale
    # dp_size or scale up world_size, i.e., edp_size
    cp = 1

    # TODO: support other orders
    order = "tp-cp-ep-dp-pp"
    # In training:
    # Dense:
    # global_rank = tp_rank + cp_rank * tp_size + dp_rank * cp_size * tp_size + pp_rank * dp_size * cp_size * tp_size
    # MoE:
    # global_rank = etp_rank + ep_rank * etp_size + edp_rank * ep_size * etp_size + pp_rank * edp_size * ep_size * etp_size

    # In ckpt loading, each rank will load a checkpoint according to its (tp_rank, pp_rank, ep_rank)
    # Thus, (tp_rank, ep_rank) should map to a unique etp_rank
    rank_mappings = dict()
    local_ids = []
    for global_rank in range(minimal_worldsize):
        tp_rank = global_rank % tp
        etp_rank = global_rank % etp
        ep_rank = (global_rank // etp) % ep
        pp_rank = global_rank // (dp * tp)

        if (tp_rank, ep_rank) not in rank_mappings:
            rank_mappings[(tp_rank, ep_rank)] = etp_rank

        if rank_mappings[(tp_rank, ep_rank)] != etp_rank:
            raise ValueError("The legacy checkpoint format cannot support this parallel config.")

        local_ids.append((tp_rank, etp_rank, ep_rank, pp_rank))
    return local_ids


def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-tensor-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    parser.add_argument(
        "--hf-max-shard-size",
        type=str,
        default="5GB"
    )

    return parser


def load_megatron_model(args):
    track_memory("LOAD_MG_START")

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)

    track_memory("AFTER_FILE_OPERATIONS")

    # CRITICAL: Update parallelism args BEFORE model creation
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    track_memory("BEFORE_MODEL_PROVIDER")
    print(f"🔍 Creating MG model with qk_layernorm={args.qk_layernorm}, tp={args.tensor_model_parallel_size}, pp={args.pipeline_model_parallel_size}")
    model = model_provider()
    track_memory("AFTER_MODEL_PROVIDER")

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels

    group_per_split = args.num_query_groups // args.tensor_model_parallel_size

    if args.num_experts is not None:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    mid_state = defaultdict(list)

    # GLOBAL REFERENCE COUNTING: Track checkpoint tensor usage for safe memory cleanup
    checkpoint_tensor_refs = {}  # tensor_id -> count
    checkpoint_tensors = {}      # tensor_id -> tensor (for cleanup)

    def register_checkpoint_tensor_usage(tensor, expected_uses=1):
        """Register expected usage count for a checkpoint tensor"""
        if isinstance(tensor, torch.Tensor):
            tensor_id = id(tensor)
            if tensor_id not in checkpoint_tensor_refs:
                checkpoint_tensor_refs[tensor_id] = 0
                checkpoint_tensors[tensor_id] = tensor
            checkpoint_tensor_refs[tensor_id] += expected_uses

    def use_and_cleanup_if_done(tensor):
        """Mark tensor as used and apply magic sauce if no more uses expected"""
        if isinstance(tensor, torch.Tensor):
            tensor_id = id(tensor)
            if tensor_id in checkpoint_tensor_refs:
                checkpoint_tensor_refs[tensor_id] -= 1
                if checkpoint_tensor_refs[tensor_id] == 0:
                    # Safe to apply magic sauce - no more uses expected
                    try:
                        tensor.untyped_storage().resize_(0)
                    except RuntimeError:
                        pass  # Skip non-resizable tensors
                    # Clean up tracking
                    del checkpoint_tensor_refs[tensor_id]
                    del checkpoint_tensors[tensor_id]

    # STREAMING APPROACH: Process and load weights directly into model to avoid accumulation
    def get_model_parameter(model, param_name):
        """Get model parameter by name without creating a full state_dict"""
        parts = param_name.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, '_parameters') and part in current._parameters:
                return current._parameters[part]
            elif hasattr(current, '_modules') and part in current._modules:
                current = current._modules[part]
            else:
                return None
        return current if isinstance(current, torch.nn.Parameter) else None

    def process_and_load_chunk_streaming(chunk_mid_state, model):
        """Process a chunk of mid_state and load directly into model, avoiding state_dict accumulation"""
        track_memory("BEFORE_CHUNK_PROCESSING")

        # Create a list of items to avoid "dictionary changed size during iteration"
        chunk_items = list(chunk_mid_state.items())
        chunk_state_dict = {}

        # CRITICAL: Track intermediate tensors for cleanup
        intermediate_tensors = []

        for k, v in chunk_items:
            if 'extra_state' in k:
                continue
            elif not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'word_embeddings' in k or 'output_layer' in k:
                # TRIED & TESTED PATTERN: Clone, create target, then immediately cleanup sources
                v_clones = [tensor.clone() for tensor in v]
                for original in v:
                    try:
                        original.untyped_storage().resize_(0)  # Magic sauce BEFORE creating target
                    except RuntimeError:
                        pass
                target_v = torch.cat(v_clones, dim=0)
                intermediate_tensors.extend(v_clones)  # Track clones for later cleanup
            elif 'linear_proj' in k:
                # TRIED & TESTED PATTERN: Clone, create target, then immediately cleanup sources
                v_clones = [tensor.clone() for tensor in v]
                for original in v:
                    try:
                        original.untyped_storage().resize_(0)  # Magic sauce BEFORE creating target
                    except RuntimeError:
                        pass
                target_v = torch.cat(v_clones, dim=1)
                intermediate_tensors.extend(v_clones)  # Track clones for later cleanup
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                cat_tensor = torch.cat(viewed, dim=0)
                intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                target_v = cat_tensor.view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                cat_tensor = torch.cat(viewed, dim=0)
                intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                target_v = cat_tensor.view(-1)
            elif "experts.linear_fc2" in k and "shared_experts" not in k:
                target_v = torch.cat(v, dim=1)
                # CRITICAL: Track source tensors for cleanup
                intermediate_tensors.extend(v)
            elif "experts.linear_fc1" in k and "shared_experts" not in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                cat_tensor = torch.cat(viewed, dim=1)
                intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                target_v = cat_tensor.view(-1, args.hidden_size)
            elif "shared_experts.gate_weight" in k or 'layer_norm_weight' in k or 'pre_mlp_layernorm' in k or 'final_layernorm' in k or 'q_layernorm' in k or 'k_layernorm' in k:
                target_v = v[0]
            else:
                print(f"⚠️  WARNING: Unhandled key {k}, using first element")
                target_v = v[0]

            chunk_state_dict[k] = target_v

            # Immediately free memory for processed tensors
            if k in chunk_mid_state:
                del chunk_mid_state[k]

        # STREAMING: Load this chunk directly into model parameters with safe cleanup
        # Use direct parameter access instead of shared_model_dict to avoid memory references

        # Store references to tensors that need cleanup after copying
        tensors_to_cleanup = []

        for param_name, param_tensor in list(chunk_state_dict.items()):
            # Get parameter directly from model without creating full state_dict
            model_param = get_model_parameter(model, param_name)
            if model_param is not None:
                # Direct parameter assignment
                model_param.copy_(param_tensor)
                # Queue tensor for cleanup AFTER copy is complete
                tensors_to_cleanup.append(param_tensor)
                del chunk_state_dict[param_name]
            else:
                print(f"⚠️  WARNING: Parameter {param_name} not found in model")
                tensors_to_cleanup.append(param_tensor)

        # SAFE CLEANUP: Apply magic sauce only after all copies are complete
        for tensor in tensors_to_cleanup:
            if isinstance(tensor, torch.Tensor):
                try:
                    tensor.untyped_storage().resize_(0)  # The magic sauce
                except RuntimeError:
                    pass  # Skip non-resizable tensors
        del tensors_to_cleanup

        # CRITICAL CLEANUP: Clean up intermediate tensors created during target_v operations
        print(f"🧹 Cleaning up {len(intermediate_tensors)} intermediate tensors")
        successful_cleanups = 0
        failed_cleanups = 0
        for tensor in intermediate_tensors:
            if isinstance(tensor, torch.Tensor):
                try:
                    storage_size_before = tensor.untyped_storage().size()
                    tensor.untyped_storage().resize_(0)  # The magic sauce
                    successful_cleanups += 1
                    if successful_cleanups <= 3:  # Log first few for debugging
                        print(f"✅ Cleaned tensor: {storage_size_before} bytes → 0")
                except RuntimeError as e:
                    failed_cleanups += 1
                    if failed_cleanups <= 3:  # Log first few failures
                        print(f"❌ Failed to clean tensor: {e}")
        print(f"🔧 Cleanup results: {successful_cleanups} successful, {failed_cleanups} failed")
        del intermediate_tensors

        # Force immediate garbage collection after magic sauce
        gc.collect()

        # AGGRESSIVE CLEANUP: Now safely clean up source tensors from chunk_mid_state
        for k, v in chunk_items:
            if isinstance(v, list):
                for tensor in v:
                    if isinstance(tensor, torch.Tensor):
                        try:
                            tensor.untyped_storage().resize_(0)  # Magic sauce on cloned tensors
                        except RuntimeError:
                            pass
                v.clear()

        del chunk_state_dict
        # Note: Using direct parameter access, no shared dict to manage

        # ULTRA AGGRESSIVE garbage collection and cache clearing
        gc.collect()
        gc.collect()  # Call twice
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python to release memory back to OS
        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Ignore if not available

        track_memory("AFTER_CHUNK_PROCESSING")

    # Configuration for chunked processing - SMALLER CHUNKS for better memory control
    CHUNK_SIZE = 200  # Process weights in chunks of 20 to avoid memory buildup

    def process_mid_state_if_full(mid_state, model, threshold=CHUNK_SIZE * 3):
        """Process mid_state immediately if it gets too large to prevent memory buildup"""
        if len(mid_state) >= threshold:
            print(f"🔍 IMMEDIATE STREAMING: Processing {len(mid_state)} items to prevent memory buildup")
            # Process first chunk
            mid_state_keys = list(mid_state.keys())[:CHUNK_SIZE]
            chunk_mid_state = {k: mid_state[k] for k in mid_state_keys}
            process_and_load_chunk_streaming(chunk_mid_state, model)
            # Remove processed keys
            for k in mid_state_keys:
                del mid_state[k]
            gc.collect()
            track_memory("AFTER_IMMEDIATE_CHUNK")

    if (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.expert_tensor_parallel_size == 1
    ):
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(tp_rank, args.expert_model_parallel_size, args.tensor_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    # Progressive loading with immediate memory cleanup
                    # if ENABLE_DEBUG:
                    #     checkpoint_name = "/workspace/mega-models/Qwen3-Coder-30B-A3B-Instruct/release/mp_rank_00/model_optim_rng.pt"
                    checkpoint_data = torch.load(checkpoint_name, map_location="cpu", weights_only=False)
                    split_state = checkpoint_data['model']

                    for orig_k in list(split_state.keys()):
                        v = split_state[orig_k]
                        if "_extra_state" in orig_k:
                            continue
                        k = orig_k
                        try:
                            if 'experts' in k and "shared_experts" not in k:
                                pattern = r'weight(\d+)'
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)

                            # Process tensor and add to mid_state
                            if 'linear_proj' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k or \
                                "linear_qkv" in k:
                                if ep_rank == tp_rank:
                                    # MEMORY OPTIMIZATION: Clone tensor for mid_state with reference tracking
                                    if isinstance(v, torch.Tensor):
                                        register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                        cloned_tensor = v.clone()
                                        mid_state[tgt].append(cloned_tensor)
                                        use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                    else:
                                        mid_state[tgt].append(v)
                            else:
                                # MEMORY OPTIMIZATION: Clone tensor for mid_state with reference tracking
                                if isinstance(v, torch.Tensor):
                                    register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                    cloned_tensor = v.clone()
                                    mid_state[tgt].append(cloned_tensor)
                                    use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                else:
                                    mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank == tp_rank and pp_rank == 0:
                                    # MEMORY OPTIMIZATION: Clone and immediately free original
                                    if isinstance(v, torch.Tensor):
                                        register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                        cloned_tensor = v.clone()
                                        mid_state[k].append(cloned_tensor)
                                        use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                    else:
                                        mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == tp_rank and pp_rank == args.pipeline_model_parallel_size - 1:
                                    # MEMORY OPTIMIZATION: Clone and immediately free original
                                    if isinstance(v, torch.Tensor):
                                        register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                        cloned_tensor = v.clone()
                                        mid_state[k].append(cloned_tensor)
                                        use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                    else:
                                        mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing!! ")

                        # MEMORY OPTIMIZATION: Remove from split_state and garbage collect
                        split_state.pop(orig_k, None)
                        del v
                        if len(split_state) % 50 == 0:  # GC every 50 tensors
                            gc.collect()

                    # MEMORY OPTIMIZATION: Final cleanup of checkpoint data
                    del checkpoint_data
                    del split_state
                    gc.collect()

                    # IMMEDIATE STREAMING: Disabled for now to avoid multiple model.state_dict() calls
                    # process_mid_state_if_full(mid_state, model)

        # STREAMING APPROACH: Process mid_state in chunks to avoid memory buildup
        track_memory("BEFORE_CHUNKED_PROCESSING")
        print(f"🔍 DEBUG: Processing {len(mid_state)} keys in chunks of {CHUNK_SIZE}")

        # MEMORY OPTIMIZATION: Use direct parameter access instead of shared_model_dict
        print("🔧 Using direct parameter access to avoid memory references...")
        track_memory("BEFORE_DIRECT_PARAMETER_ACCESS")

        # Process mid_state in chunks
        mid_state_keys = list(mid_state.keys())
        for i in range(0, len(mid_state_keys), CHUNK_SIZE):
            chunk_keys = mid_state_keys[i:i + CHUNK_SIZE]
            chunk_mid_state = {k: mid_state[k] for k in chunk_keys}

            print(f"🔍 Processing chunk {i//CHUNK_SIZE + 1}/{(len(mid_state_keys) + CHUNK_SIZE - 1)//CHUNK_SIZE} ({len(chunk_keys)} keys)")
            process_and_load_chunk_streaming(chunk_mid_state, model)

            # Remove processed keys from mid_state to free memory
            for k in chunk_keys:
                del mid_state[k]

            # Aggressive garbage collection after each chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FINAL CLEANUP: Aggressive garbage collection
        gc.collect()

        track_memory("AFTER_CHUNKED_PROCESSING")

    elif (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.expert_tensor_parallel_size > 1
    ):
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    # Progressive loading with immediate memory cleanup
                    checkpoint_data = torch.load(checkpoint_name, map_location="cpu", weights_only=False)
                    split_state = checkpoint_data['model']

                    for orig_k in list(split_state.keys()):
                        v = split_state[orig_k]
                        if "_extra_state" in orig_k:
                            continue
                        k = orig_k
                        try:
                            if 'experts' in k and "shared_experts" not in k:
                                pattern = r'weight(\d+)'
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)

                            # Process tensor and add to mid_state
                            if 'linear_proj' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k or \
                                "linear_qkv" in k:
                                if ep_rank == 0:
                                    # MEMORY OPTIMIZATION: Clone and immediately free original
                                    if isinstance(v, torch.Tensor):
                                        cloned_tensor = v.clone()
                                        mid_state[tgt].append(cloned_tensor)
                                    else:
                                        mid_state[tgt].append(v)
                            else:
                                # MEMORY OPTIMIZATION: Clone tensor for mid_state with reference tracking
                                if isinstance(v, torch.Tensor):
                                    register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                    cloned_tensor = v.clone()
                                    mid_state[tgt].append(cloned_tensor)
                                    use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                else:
                                    mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank == 0 and pp_rank == 0:
                                    # MEMORY OPTIMIZATION: Clone and immediately free original
                                    if isinstance(v, torch.Tensor):
                                        register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                        cloned_tensor = v.clone()
                                        mid_state[k].append(cloned_tensor)
                                        use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                    else:
                                        mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == 0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    # MEMORY OPTIMIZATION: Clone and immediately free original
                                    if isinstance(v, torch.Tensor):
                                        register_checkpoint_tensor_usage(v, 1)  # Track original tensor usage
                                        cloned_tensor = v.clone()
                                        mid_state[k].append(cloned_tensor)
                                        use_and_cleanup_if_done(v)  # Mark as used and cleanup if done
                                    else:
                                        mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing!! ")

                        # MEMORY OPTIMIZATION: Remove from split_state and garbage collect
                        split_state.pop(orig_k, None)
                        del v
                        if len(split_state) % 50 == 0:  # GC every 50 tensors
                            gc.collect()

                    # MEMORY OPTIMIZATION: Final cleanup of checkpoint data
                    del checkpoint_data
                    del split_state
                    gc.collect()

                    # IMMEDIATE STREAMING: Disabled for now to avoid multiple model.state_dict() calls
                    # process_mid_state_if_full(mid_state, model)

        # STREAMING APPROACH: Process mid_state in chunks to avoid memory buildup (Branch 2)
        track_memory("BEFORE_CHUNKED_PROCESSING_B2")
        print(f"🔍 DEBUG: Processing {len(mid_state)} keys in chunks of {CHUNK_SIZE} (Branch 2)")

        # Process mid_state in chunks for the second branch
        mid_state_keys = list(mid_state.keys())
        for i in range(0, len(mid_state_keys), CHUNK_SIZE):
            chunk_keys = mid_state_keys[i:i + CHUNK_SIZE]
            chunk_mid_state = {k: mid_state[k] for k in chunk_keys}

            print(f"🔍 Processing chunk {i//CHUNK_SIZE + 1}/{(len(mid_state_keys) + CHUNK_SIZE - 1)//CHUNK_SIZE} ({len(chunk_keys)} keys)")

            # Process chunk with branch-2 specific logic and stream directly to model
            chunk_state_dict = {}

            # CRITICAL: Track intermediate tensors for cleanup
            intermediate_tensors = []

            for k, v in chunk_mid_state.items():
                if 'extra_state' in k:
                    continue
                elif not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                    target_v = v[0]
                elif 'word_embeddings' in k or 'output_layer' in k:
                    target_v = torch.cat(v, dim=0)
                    # CRITICAL: Track source tensors for cleanup
                    intermediate_tensors.extend(v)
                elif 'linear_proj' in k:
                    target_v = torch.cat(v, dim=1)
                    # CRITICAL: Track source tensors for cleanup
                    intermediate_tensors.extend(v)
                elif 'linear_qkv.weight' in k:
                    viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                    intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                    intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                    cat_tensor = torch.cat(viewed, dim=0)
                    intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                    target_v = cat_tensor.view(-1, args.hidden_size)
                elif 'linear_qkv.bias' in k:
                    viewed = [x.view(group_per_split, -1) for x in v]
                    intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                    intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                    cat_tensor = torch.cat(viewed, dim=0)
                    intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                    target_v = cat_tensor.view(-1)
                elif 'linear_fc1' in k:
                    viewed = [x.view(2, -1, args.hidden_size) for x in v]
                    intermediate_tensors.extend(viewed)  # Track intermediate view tensors
                    intermediate_tensors.extend(v)  # CRITICAL: Also track original source tensors
                    cat_tensor = torch.cat(viewed, dim=1)
                    intermediate_tensors.append(cat_tensor)  # Track intermediate cat tensor
                    target_v = cat_tensor.view(-1, args.hidden_size)
                elif "linear_fc2" in k:
                    target_v = torch.cat(v, dim=1)
                    # CRITICAL: Track source tensors for cleanup
                    intermediate_tensors.extend(v)
                elif "shared_experts.gate_weight" in k or 'layer_norm_weight' in k or 'pre_mlp_layernorm' in k or 'final_layernorm' in k or 'q_layernorm' in k or 'k_layernorm' in k:
                    target_v = v[0]
                else:
                    print(f"⚠️  WARNING: Unhandled key {k}, using first element")
                    target_v = v[0]

                chunk_state_dict[k] = target_v

            # STREAMING: Load this chunk directly into model parameters with safe cleanup
            # Use direct parameter access instead of shared_model_dict to avoid memory references

            # Store references to tensors that need cleanup after copying
            tensors_to_cleanup = []

            for param_name, param_tensor in list(chunk_state_dict.items()):
                # Get parameter directly from model without creating full state_dict
                model_param = get_model_parameter(model, param_name)
                if model_param is not None:
                    # Direct parameter assignment
                    model_param.copy_(param_tensor)
                    # Queue tensor for cleanup AFTER copy is complete
                    tensors_to_cleanup.append(param_tensor)
                    del chunk_state_dict[param_name]
                else:
                    print(f"⚠️  WARNING: Parameter {param_name} not found in model")
                    tensors_to_cleanup.append(param_tensor)

            # SAFE CLEANUP: Apply magic sauce only after all copies are complete
            for tensor in tensors_to_cleanup:
                if isinstance(tensor, torch.Tensor):
                    try:
                        tensor.untyped_storage().resize_(0)  # The magic sauce
                    except RuntimeError:
                        pass  # Skip non-resizable tensors
            del tensors_to_cleanup

            # CRITICAL CLEANUP: Clean up intermediate tensors created during target_v operations
            print(f"🧹 Cleaning up {len(intermediate_tensors)} intermediate tensors")
            successful_cleanups = 0
            failed_cleanups = 0
            for tensor in intermediate_tensors:
                if isinstance(tensor, torch.Tensor):
                    try:
                        storage_size_before = tensor.untyped_storage().size()
                        tensor.untyped_storage().resize_(0)  # The magic sauce
                        successful_cleanups += 1
                        if successful_cleanups <= 3:  # Log first few for debugging
                            print(f"✅ Cleaned tensor: {storage_size_before} bytes → 0")
                    except RuntimeError as e:
                        failed_cleanups += 1
                        if failed_cleanups <= 3:  # Log first few failures
                            print(f"❌ Failed to clean tensor: {e}")
            print(f"🔧 Cleanup results: {successful_cleanups} successful, {failed_cleanups} failed")
            del intermediate_tensors

            del chunk_state_dict

            # Force immediate garbage collection after magic sauce
            gc.collect()

            # AGGRESSIVE CLEANUP: Now safely clean up source tensors from chunk_mid_state
            chunk_items = list(chunk_mid_state.items())
            for k, v in chunk_items:
                if isinstance(v, list):
                    for tensor in v:
                        if isinstance(tensor, torch.Tensor):
                            try:
                                tensor.untyped_storage().resize_(0)  # Magic sauce on source tensors
                            except RuntimeError:
                                pass
                    v.clear()

            # Remove processed keys from mid_state to free memory
            for k in chunk_keys:
                del mid_state[k]

            # ULTRA AGGRESSIVE garbage collection and cache clearing
            gc.collect()
            gc.collect()  # Call twice
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force Python to release memory back to OS
            import ctypes
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass  # Ignore if not available

        track_memory("AFTER_CHUNKED_PROCESSING_B2")

    else:
        raise ValueError('not support yet')

    # STREAMING APPROACH: Weights are already loaded directly into model during chunked processing
    track_memory("AFTER_STREAMING_LOAD")
    print("✅ Streaming checkpoint loading completed - all weights loaded directly into model")

    # Print memory usage summary
    print("\n🔍 MEMORY USAGE SUMMARY:")
    for entry in _memory_history:
        if len(entry) == 2:  # Old format: (location, memory_gb)
            location, memory_gb = entry
            print(f"  {location}: {memory_gb:.2f} GB")
        elif len(entry) == 4:  # New format: (location, memory_gb, hf_size, mg_size)
            location, memory_gb, hf_size, mg_size = entry
            total_models = hf_size + mg_size
            hf_indicator = "meta✓" if hf_size == 0.0 else f"{hf_size:.2f}"
            print(f"  {location}: System={memory_gb:.2f}GB | HF={hf_indicator} | MG={mg_size:.2f} | Total={total_models:.2f}")

    return model



def convert_checkpoint_from_megatron_to_transformers_inplace(mgmodel, hfmodel, args):
    """Assign Megatron weights directly into HF model without large state_dict materialization."""
    track_memory_with_models("B_INPLACE_CONV_START", hfmodel, mgmodel)

    target_dtype = getattr(args, "torch_dtype", None)
    if isinstance(target_dtype, str):
        target_dtype = getattr(torch, target_dtype)
    if target_dtype is None:
        target_dtype = mgmodel.embedding.word_embeddings.weight.dtype

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim

    with torch.no_grad():
        emb = _to_cpu_dtype(mgmodel.embedding.word_embeddings.weight, target_dtype)
        _assign_param_by_name(hfmodel, "model.embed_tokens.weight", emb)
        _free_storage_safe(mgmodel.embedding.word_embeddings.weight)
        del emb
        gc.collect()

        for layer_idx, mglayer in enumerate(mgmodel.decoder.layers):
            if ENABLE_DEBUG and layer_idx == 0:
                track_memory(f"INPLACE_L{layer_idx}_START")

            if use_te:
                src = mglayer.self_attention.linear_qkv.layer_norm_weight
            else:
                src = mglayer.input_layernorm.weight
            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.input_layernorm.weight",
                _to_cpu_dtype(src, target_dtype),
            )
            _free_storage_safe(src)

            w = mglayer.self_attention.linear_qkv.weight
            qkv = w.view(num_query_groups, -1, head_dim, hidden_size)

            q_w = qkv[:, :value_num_per_group].reshape(-1, hidden_size)
            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                _to_cpu_dtype(q_w, target_dtype),
            )

            k_w = qkv[:, value_num_per_group:value_num_per_group + 1].reshape(-1, hidden_size)
            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                _to_cpu_dtype(k_w, target_dtype),
            )

            v_w = qkv[:, value_num_per_group + 1:value_num_per_group + 2].reshape(-1, hidden_size)
            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                _to_cpu_dtype(v_w, target_dtype),
            )

            _free_storage_safe(w)
            del qkv, q_w, k_w, v_w
            gc.collect()

            if args.add_qkv_bias:
                b = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)

                q_b = b[:, :q_dim_per_group].contiguous().view(-1)
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.self_attn.q_proj.bias",
                    _to_cpu_dtype(q_b, target_dtype),
                )

                k_b = b[:, q_dim_per_group:q_dim_per_group + kv_dim_per_group].contiguous().view(-1)
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.self_attn.k_proj.bias",
                    _to_cpu_dtype(k_b, target_dtype),
                )

                v_b = b[:, q_dim_per_group + kv_dim_per_group:q_dim_per_group + 2 * kv_dim_per_group].contiguous().view(-1)
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.self_attn.v_proj.bias",
                    _to_cpu_dtype(v_b, target_dtype),
                )

                _free_storage_safe(mglayer.self_attention.linear_qkv.bias)
                del b, q_b, k_b, v_b
                gc.collect()

            if args.qk_layernorm:
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                    _to_cpu_dtype(mglayer.self_attention.q_layernorm.weight, target_dtype),
                )
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.self_attn.k_norm.weight",
                    _to_cpu_dtype(mglayer.self_attention.k_layernorm.weight, target_dtype),
                )
                _free_storage_safe(mglayer.self_attention.q_layernorm.weight)
                _free_storage_safe(mglayer.self_attention.k_layernorm.weight)

            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                _to_cpu_dtype(mglayer.self_attention.linear_proj.weight, target_dtype),
            )
            _free_storage_safe(mglayer.self_attention.linear_proj.weight)

            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.mlp.gate.weight",
                _to_cpu_dtype(mglayer.mlp.router.weight, target_dtype),
            )
            _free_storage_safe(mglayer.mlp.router.weight)

            if args.num_experts is None:
                raise ValueError("num_experts is None")

            hflayer = hfmodel.model.layers[layer_idx]
            experts = hflayer.mlp.experts
            use_packed_experts = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")

            if use_packed_experts:
                # Ensure packed expert tensors are materialized on CPU if still on meta.
                if experts.gate_up_proj.device.type == "meta":
                    gate_up_cpu = torch.empty(
                        experts.gate_up_proj.shape, device="cpu", dtype=target_dtype
                    )
                    _assign_param_by_name(
                        hfmodel,
                        f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                        gate_up_cpu,
                    )
                if experts.down_proj.device.type == "meta":
                    down_cpu = torch.empty(
                        experts.down_proj.shape, device="cpu", dtype=target_dtype
                    )
                    _assign_param_by_name(
                        hfmodel,
                        f"model.layers.{layer_idx}.mlp.experts.down_proj",
                        down_cpu,
                    )

                # Refresh references after possible replacement.
                experts = hfmodel.model.layers[layer_idx].mlp.experts

                for expert_idx in range(args.num_experts):
                    fc1 = getattr(mglayer.mlp.experts.linear_fc1, f"weight{expert_idx}")
                    experts.gate_up_proj[expert_idx].copy_(_to_cpu_dtype(fc1, target_dtype))
                    _free_storage_safe(fc1)
                    del fc1

                    fc2 = getattr(mglayer.mlp.experts.linear_fc2, f"weight{expert_idx}")
                    experts.down_proj[expert_idx].copy_(_to_cpu_dtype(fc2, target_dtype))
                    _free_storage_safe(fc2)
                    del fc2
            else:
                for expert_idx in range(args.num_experts):
                    fc1 = getattr(mglayer.mlp.experts.linear_fc1, f"weight{expert_idx}")
                    gate_w, up_w = torch.split(fc1, split_size_or_sections=args.moe_ffn_hidden_size, dim=0)

                    _assign_param_by_name(
                        hfmodel,
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        _to_cpu_dtype(gate_w, target_dtype),
                    )
                    _assign_param_by_name(
                        hfmodel,
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        _to_cpu_dtype(up_w, target_dtype),
                    )

                    _free_storage_safe(fc1)
                    del gate_w, up_w

                    fc2 = getattr(mglayer.mlp.experts.linear_fc2, f"weight{expert_idx}")
                    _assign_param_by_name(
                        hfmodel,
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        _to_cpu_dtype(fc2, target_dtype),
                    )
                    _free_storage_safe(fc2)
                    del fc2

            if args.moe_shared_expert_intermediate_size is not None:
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
                    _to_cpu_dtype(mglayer.mlp.shared_experts.gate_weight, target_dtype),
                )
                _free_storage_safe(mglayer.mlp.shared_experts.gate_weight)

                se_fc1 = mglayer.mlp.shared_experts.linear_fc1.weight
                se_gate, se_up = torch.split(
                    se_fc1,
                    split_size_or_sections=args.moe_shared_expert_intermediate_size,
                    dim=0,
                )
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight",
                    _to_cpu_dtype(se_gate, target_dtype),
                )
                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
                    _to_cpu_dtype(se_up, target_dtype),
                )
                _free_storage_safe(se_fc1)
                del se_gate, se_up

                _assign_param_by_name(
                    hfmodel,
                    f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
                    _to_cpu_dtype(mglayer.mlp.shared_experts.linear_fc2.weight, target_dtype),
                )
                _free_storage_safe(mglayer.mlp.shared_experts.linear_fc2.weight)
                del se_fc1

            _assign_param_by_name(
                hfmodel,
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                _to_cpu_dtype(mglayer.pre_mlp_layernorm.weight, target_dtype),
            )
            _free_storage_safe(mglayer.pre_mlp_layernorm.weight)

            cleanup_mg_layer_comprehensive(mglayer)
            gc.collect()

            if ENABLE_DEBUG and (layer_idx + 1) % 5 == 0:
                track_memory(f"INPLACE_AFTER_L{layer_idx + 1}")

        _assign_param_by_name(
            hfmodel,
            "model.norm.weight",
            _to_cpu_dtype(mgmodel.decoder.final_layernorm.weight, target_dtype),
        )
        _free_storage_safe(mgmodel.decoder.final_layernorm.weight)

        if getattr(mgmodel, "output_layer", None) is not None:
            _assign_param_by_name(
                hfmodel,
                "lm_head.weight",
                _to_cpu_dtype(mgmodel.output_layer.weight, target_dtype),
            )
            _free_storage_safe(mgmodel.output_layer.weight)

        meta_init_count = 0
        for name, buf in hfmodel.named_buffers():
            if buf.device.type == "meta":
                _assign_buffer_by_name(
                    hfmodel,
                    name,
                    torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                )
                meta_init_count += 1
        if meta_init_count:
            print(f"✅ Initialized {meta_init_count} meta buffers on CPU")

    track_memory_with_models("B_INPLACE_CONV_END", hfmodel, mgmodel)
    _assert_no_meta(hfmodel)
    return hfmodel


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
    """Convert Megatron model to HuggingFace format using memory-efficient state_dict approach.

    This function builds a state_dict from the MG model and applies it to the HF model
    using assign=True for optimal memory efficiency.
    """
    # MEMORY DEBUGGING: Track memory at conversion start
    track_memory_with_models("CONVERSION_FUNCTION_START", hfmodel, mgmodel)

    if args.fp16:
        mgmodel = mgmodel.half()
        # Note: HF model stays on meta device, dtype will be set during state_dict materialization
        track_memory_with_models("AFTER_FP16_CONVERSION", hfmodel, mgmodel)
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        # Note: HF model stays on meta device, dtype will be set during state_dict materialization
        track_memory_with_models("AFTER_BF16_CONVERSION", hfmodel, mgmodel)

    # BUILD STATE_DICT: Create HF state_dict from MG model parameters
    print("🔄 Building HuggingFace state_dict from Megatron model...")
    hf_state_dict = build_hf_state_dict_from_mg_model(mgmodel, args)

    # MATERIALIZE MODEL: Apply state_dict to HF model with assign=True for memory efficiency
    print("🔄 Materializing HuggingFace model from state_dict with assign=True...")
    materialize_hf_model_from_state_dict(hfmodel, hf_state_dict, args)

    track_memory_with_models("CONVERSION_FUNCTION_END", hfmodel, mgmodel)
    return hfmodel

def build_hf_state_dict_from_mg_model(mgmodel, args):
    """Build HuggingFace state_dict from Megatron model parameters.

    This function extracts and transforms all parameters from the MG model
    into the HF format without direct copying to save memory.
    """
    hf_state_dict = {}

    # Calculate dimensions for weight transformations
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups  # 28//4
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim

    with torch.no_grad():
        track_memory("STATE_DICT_BUILD_START")

        # EMBEDDING WEIGHTS
        print("🔧 Building embedding weights...")
        hf_state_dict['model.embed_tokens.weight'] = mgmodel.embedding.word_embeddings.weight.clone()

        # MAGIC SAUCE: Immediately free MG embedding after cloning
        try:
            mgmodel.embedding.word_embeddings.weight.untyped_storage().resize_(0)
        except RuntimeError:
            pass

        track_memory("AFTER_EMBEDDING_STATE_DICT")

        # LAYER WEIGHTS
        print(f"🔧 Building state_dict for {len(mgmodel.decoder.layers)} layers...")
        for layer_idx, mglayer in enumerate(mgmodel.decoder.layers):
            if layer_idx == 0:
                track_memory("FIRST_LAYER_STATE_DICT_START")

            print(f"🔄 Building state_dict for layer {layer_idx+1}/{len(mgmodel.decoder.layers)}")

            # INPUT LAYERNORM
            if use_te:
                hf_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
                    mglayer.self_attention.linear_qkv.layer_norm_weight.clone()
            else:
                hf_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
                    mglayer.input_layernorm.weight.clone()

            # QKV PROCESSING WITH MAGIC SAUCE PATTERN
            if ENABLE_DEBUG:
                print(f"  🔧 Processing QKV weights...")

            # Step 1: Clone the source weight
            qkv_weight_clone = mglayer.self_attention.linear_qkv.weight.clone()

            # Step 2: Apply magic sauce to original immediately
            try:
                mglayer.self_attention.linear_qkv.weight.untyped_storage().resize_(0)
            except RuntimeError:
                pass

            # Step 3: Process using clone
            qkv_weight = qkv_weight_clone.view(num_query_groups, -1, head_dim, hidden_size)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)

            # Create final tensors and assign to state_dict
            hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = q_weight.reshape(-1, hidden_size).clone()
            hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = k_weight.reshape(-1, hidden_size).clone()
            hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = v_weight.reshape(-1, hidden_size).clone()

            # Step 4: Clean up clones and intermediates
            try:
                qkv_weight_clone.untyped_storage().resize_(0)
                qkv_weight.untyped_storage().resize_(0)
                q_weight.untyped_storage().resize_(0)
                k_weight.untyped_storage().resize_(0)
                v_weight.untyped_storage().resize_(0)
            except RuntimeError:
                pass

            # Step 5: Force GC
            del qkv_weight_clone, qkv_weight, q_weight, k_weight, v_weight
            gc.collect()

            # QKV BIAS (if enabled)
            if args.add_qkv_bias:
                # MAGIC SAUCE PATTERN: Process bias tensors with cleanup
                qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)
                q_bias_split, k_bias_split, v_bias_split = torch.split(qkv_bias, split_size_or_sections=[q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1)

                # Create final bias tensors
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.bias'] = q_bias_split.contiguous().view(-1).clone()
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.bias'] = k_bias_split.contiguous().view(-1).clone()
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.bias'] = v_bias_split.contiguous().view(-1).clone()

                # MAGIC SAUCE: Clean up intermediate bias tensors
                try:
                    qkv_bias.untyped_storage().resize_(0)
                    q_bias_split.untyped_storage().resize_(0)
                    k_bias_split.untyped_storage().resize_(0)
                    v_bias_split.untyped_storage().resize_(0)
                except RuntimeError:
                    pass  # Continue if resize fails

                del qkv_bias, q_bias_split, k_bias_split, v_bias_split
                gc.collect()

            # QK LAYERNORM (if enabled)
            if args.qk_layernorm:
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_norm.weight'] = \
                    mglayer.self_attention.q_layernorm.weight.data.clone()
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_norm.weight'] = \
                    mglayer.self_attention.k_layernorm.weight.data.clone()

            # OUTPUT PROJECTION
            hf_state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = \
                mglayer.self_attention.linear_proj.weight.clone()

            # ROUTER/GATE WEIGHT
            if args.num_experts is None:
                raise ValueError("num_experts is None")

            hf_state_dict[f'model.layers.{layer_idx}.mlp.gate.weight'] = \
                mglayer.mlp.router.weight.clone()

            # EXPERTS PROCESSING WITH MAGIC SAUCE PATTERN
            if ENABLE_DEBUG:
                print(f"  🔧 Processing {args.num_experts} experts...")

            for i in range(args.num_experts):
                linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))

                # Split fc1 weight into gate and up weights
                gate_weight, up_weight = torch.split(linear_fc1_weighti,
                                                        split_size_or_sections=args.moe_ffn_hidden_size)

                # Assign to state_dict
                hf_state_dict[f'model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight'] = gate_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight'] = up_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight'] = linear_fc2_weighti.clone()

                # MAGIC SAUCE: Clean up split tensors for this expert
                try:
                    gate_weight.untyped_storage().resize_(0)
                    up_weight.untyped_storage().resize_(0)
                except RuntimeError:
                    pass  # Skip non-resizable tensors

                del gate_weight, up_weight

            # SHARED EXPERTS (if enabled)
            if args.moe_shared_expert_intermediate_size is not None:
                if ENABLE_DEBUG:
                    print(f"  🔧 Processing shared expert...")

                hf_state_dict[f'model.layers.{layer_idx}.mlp.shared_expert_gate.weight'] = \
                    mglayer.mlp.shared_experts.gate_weight.clone()

                # Split shared expert fc1 weight
                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size)

                # Assign to state_dict
                hf_state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight'] = shared_expert_gate_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight'] = shared_expert_up_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight'] = \
                    mglayer.mlp.shared_experts.linear_fc2.weight.clone()

                # MAGIC SAUCE: Clean up shared expert split tensors
                try:
                    shared_expert_gate_weight.untyped_storage().resize_(0)
                    shared_expert_up_weight.untyped_storage().resize_(0)
                except RuntimeError:
                    pass  # Skip non-resizable tensors

                del shared_expert_gate_weight, shared_expert_up_weight
                gc.collect()

            # POST ATTENTION LAYERNORM
            hf_state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = \
                mglayer.pre_mlp_layernorm.weight.clone()

            # COMPLETE MG LAYER CLEANUP: Systematically delete ALL tensors in this layer
            cleanup_mg_layer_comprehensive(mglayer)

            if layer_idx == 0:
                track_memory("FIRST_LAYER_STATE_DICT_END")

            # Periodic garbage collection
            if (layer_idx + 1) % 5 == 0:  # Every 5 layers
                gc.collect()
                track_memory(f"AFTER_LAYER_{layer_idx+1}_STATE_DICT")

        # FINAL LAYERS (norm and lm_head)
        print("🔧 Building final layer weights...")

        # Final layer norm
        hf_state_dict['model.norm.weight'] = mgmodel.decoder.final_layernorm.weight.clone()

        # LM head (output embeddings)
        if hasattr(mgmodel, 'output_layer') and mgmodel.output_layer is not None:
            hf_state_dict['lm_head.weight'] = mgmodel.output_layer.weight.clone()
        else:
            # Tied embeddings - use the same as input embeddings
            hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight'].clone()

        track_memory("STATE_DICT_BUILD_COMPLETE")

        print(f"✅ Built HuggingFace state_dict with {len(hf_state_dict)} parameters")
        return hf_state_dict

def cleanup_mg_layer_comprehensive(layer):
    """Apply magic sauce to all tensors in an MG layer"""
    try:
        # Self-attention cleanup
        if hasattr(layer.self_attention, 'linear_qkv'):
            if hasattr(layer.self_attention.linear_qkv, 'weight'):
                layer.self_attention.linear_qkv.weight.untyped_storage().resize_(0)
            if hasattr(layer.self_attention.linear_qkv, 'bias') and layer.self_attention.linear_qkv.bias is not None:
                layer.self_attention.linear_qkv.bias.untyped_storage().resize_(0)
            if hasattr(layer.self_attention.linear_qkv, 'layer_norm_weight'):
                layer.self_attention.linear_qkv.layer_norm_weight.untyped_storage().resize_(0)

        if hasattr(layer.self_attention, 'linear_proj') and hasattr(layer.self_attention.linear_proj, 'weight'):
            layer.self_attention.linear_proj.weight.untyped_storage().resize_(0)

        # Q/K layernorm cleanup
        if hasattr(layer.self_attention, 'q_layernorm') and hasattr(layer.self_attention.q_layernorm, 'weight'):
            layer.self_attention.q_layernorm.weight.untyped_storage().resize_(0)
        if hasattr(layer.self_attention, 'k_layernorm') and hasattr(layer.self_attention.k_layernorm, 'weight'):
            layer.self_attention.k_layernorm.weight.untyped_storage().resize_(0)

        # MLP cleanup - router
        if hasattr(layer.mlp, 'router') and hasattr(layer.mlp.router, 'weight'):
            layer.mlp.router.weight.untyped_storage().resize_(0)

        # MLP cleanup - experts
        if hasattr(layer.mlp, 'experts'):
            if hasattr(layer.mlp.experts, 'linear_fc1'):
                for attr_name in dir(layer.mlp.experts.linear_fc1):
                    if attr_name.startswith('weight'):
                        weight_tensor = getattr(layer.mlp.experts.linear_fc1, attr_name)
                        if isinstance(weight_tensor, torch.Tensor):
                            weight_tensor.untyped_storage().resize_(0)

            if hasattr(layer.mlp.experts, 'linear_fc2'):
                for attr_name in dir(layer.mlp.experts.linear_fc2):
                    if attr_name.startswith('weight'):
                        weight_tensor = getattr(layer.mlp.experts.linear_fc2, attr_name)
                        if isinstance(weight_tensor, torch.Tensor):
                            weight_tensor.untyped_storage().resize_(0)

        # MLP cleanup - shared experts
        if hasattr(layer.mlp, 'shared_experts'):
            if hasattr(layer.mlp.shared_experts, 'gate_weight'):
                layer.mlp.shared_experts.gate_weight.untyped_storage().resize_(0)
            if hasattr(layer.mlp.shared_experts, 'linear_fc1') and hasattr(layer.mlp.shared_experts.linear_fc1, 'weight'):
                layer.mlp.shared_experts.linear_fc1.weight.untyped_storage().resize_(0)
            if hasattr(layer.mlp.shared_experts, 'linear_fc2') and hasattr(layer.mlp.shared_experts.linear_fc2, 'weight'):
                layer.mlp.shared_experts.linear_fc2.weight.untyped_storage().resize_(0)

        # Layernorm cleanup
        if hasattr(layer, 'input_layernorm') and hasattr(layer.input_layernorm, 'weight'):
            layer.input_layernorm.weight.untyped_storage().resize_(0)
        if hasattr(layer, 'pre_mlp_layernorm') and hasattr(layer.pre_mlp_layernorm, 'weight'):
            layer.pre_mlp_layernorm.weight.untyped_storage().resize_(0)

    except (RuntimeError, AttributeError) as e:
        if ENABLE_DEBUG:
            print(f"    ⚠️  Layer cleanup warning: {e}")
        pass  # Continue even if some cleanup fails

def materialize_hf_model_from_state_dict(hfmodel, hf_state_dict, args):
    """Materialize HF model from state_dict using assign=True for memory efficiency."""
    track_memory("BEFORE_STATE_DICT_MATERIALIZATION")

    print(f"🔄 Applying state_dict to HuggingFace model with assign=True...")

    # Apply state_dict with assign=True for memory efficiency
    hfmodel.load_state_dict(hf_state_dict, assign=True, strict=False)

    # Move any remaining meta tensors to CPU
    print("🔄 Moving any remaining meta tensors to CPU...")
    meta_tensor_count = 0
    for name, param in hfmodel.named_parameters():
        if param.device.type == 'meta':
            # Initialize with zeros and move to CPU, preserving dtype
            dtype = args.torch_dtype if hasattr(args, 'torch_dtype') else torch.float16
            new_param = torch.zeros(param.shape, dtype=dtype, device='cpu')
            param.data = new_param
            meta_tensor_count += 1
            if ENABLE_DEBUG:
                print(f"  🔧 Moved param {name} from meta to CPU")

    for name, buffer in hfmodel.named_buffers():
        if buffer.device.type == 'meta':
            # Initialize with zeros and move to CPU, preserving dtype
            try:
                dtype = buffer.dtype
                new_buffer = torch.zeros(buffer.shape, dtype=dtype, device='cpu')
                # Use register_buffer to properly replace the buffer
                module_names = name.split('.')
                module = hfmodel
                for module_name in module_names[:-1]:
                    module = getattr(module, module_name)
                buffer_name = module_names[-1]
                module.register_buffer(buffer_name, new_buffer)
                meta_tensor_count += 1
                if ENABLE_DEBUG:
                    print(f"  🔧 Moved buffer {name} from meta to CPU")
            except Exception as e:
                if ENABLE_DEBUG:
                    print(f"  ⚠️  Could not move buffer {name}: {e}")

    if meta_tensor_count > 0:
        print(f"✅ Moved {meta_tensor_count} tensors from meta device to CPU")
    else:
        print("✅ No meta tensors found - all parameters properly materialized")

    # Clean up state_dict to free memory
    del hf_state_dict
    gc.collect()

    track_memory("AFTER_STATE_DICT_MATERIALIZATION")
    print("✅ HuggingFace model materialization complete")

def convert_checkpoint_from_transformers_to_megatron(hfmodel: Qwen2MoeForCausalLM, mgmodel: GPTModel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    head_dim = hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        num_query_groups = args.num_query_groups

        from tqdm import tqdm
        for layer_idx, (mglayer, hflayer) in tqdm(enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)), total=len(mgmodel.decoder.layers)):

            mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hflayer.input_layernorm.weight)

            q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, args.hidden_size).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv_proj)

            if args.add_qkv_bias:
                # NOTE: compatability for Qwen3-moe
                q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
                k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
                v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
                qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
                mglayer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            if args.qk_layernorm:
                # NOTE: compatability for Qwen3-moe
                mglayer.self_attention.q_layernorm.weight.copy_(hflayer.self_attn.q_norm.weight.data)
                mglayer.self_attention.k_layernorm.weight.copy_(hflayer.self_attn.k_norm.weight.data)

            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)

            mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
            for i, hf_expert in enumerate(hflayer.mlp.experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc1_weighti.copy_(fc1_weight)
                linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                linear_fc2_weighti.copy_(hf_expert.down_proj.weight)

            if args.moe_shared_expert_intermediate_size is not None:
                # NOTE: compatability for Qwen3-moe
                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_expert.gate_proj.weight, hflayer.mlp.shared_expert.up_proj.weight])
                mglayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_experts.linear_fc2.weight.copy_(hflayer.mlp.shared_expert.down_proj.weight)
                mglayer.mlp.shared_experts.gate_weight.data.copy_(hflayer.mlp.shared_expert_gate.weight)

            mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def split_column_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[0] // tp_size
    return tensor[seg * tp_rank: seg * (tp_rank + 1)]


def split_row_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[1] // tp_size
    return tensor[:, seg * tp_rank: seg * (tp_rank + 1)]


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()


def save_mgmodel(mgmodel: GPTModel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/merges.txt " + args.save)
    os.system("cp -rf " + args.load + "/vocab.json " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size

    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if full_model[k] is None and '_extra_state' not in k:
            full_model.pop(k)
            continue
        if '_extra_state' in k and isinstance(full_model[k], torch.Tensor):
            full_model[k] = None

    if args.num_experts is not None:
        pattern = r'weight(\d+)'
        assert args.num_experts % args.expert_model_parallel_size == 0
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if args.target_decoder_first_pipeline_num_layers is not None:
        remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size - 1
        assert remained_layers % remained_stages == 0
        pp_layers_per_stage = [ args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
    else:
        pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

    for (tp_rank, etp_rank, ep_rank, pp_rank) in generate_rank_group(
            args.tensor_model_parallel_size,
            args.expert_tensor_parallel_size,
            args.expert_model_parallel_size,
            args.pipeline_model_parallel_size
    ):
        model_split = {}
        layer_offset = sum(pp_layers_per_stage[:pp_rank])
        layers_to_copy = {}
        for layer in range(pp_layers_per_stage[pp_rank]):
            pp_layer_id = layer + layer_offset
            layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
        checkpoint_name = get_checkpoint_name(
            args.save, 0, True,
            args.pipeline_model_parallel_size > 1,
            tp_rank,
            pp_rank,
            args.expert_model_parallel_size > 1,
            ep_rank
        )
        print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
        for k, v in full_model.items():
            if check_layer(layers_to_copy, k):
                layer_pattern = re.compile(r'\d+')
                res = layer_pattern.findall(k)
                k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
            elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                continue

            if not isinstance(v, torch.Tensor):
                target_v = v
            elif 'linear_qkv.weight' in k:
                viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                target_v = viewed.view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = v.view(args.num_query_groups, -1, head_dim)
                viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                target_v = viewed.view(-1)
            elif 'linear_proj' in k:
                seg = v.shape[1] // args.tensor_model_parallel_size
                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
            elif 'experts' in k and 'shared_experts' not in k:
                expert_rank = int(re.findall(pattern, k)[0])
                if expert_rank // num_local_experts != ep_rank:
                    continue
                expert_local_rank = expert_rank % num_local_experts
                k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                    seg = args.moe_ffn_hidden_size // args.expert_tensor_parallel_size
                    target_v = viewed[:, seg * etp_rank: seg * (etp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    target_v = split_row_parallel(v, etp_rank, args.expert_tensor_parallel_size)
                else:
                    raise NotImplementedError
            elif 'shared_experts' in k and 'gate' not in k:
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_shared_expert_intermediate_size,
                                    args.hidden_size)
                    seg = args.moe_shared_expert_intermediate_size // args.tensor_model_parallel_size
                    target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    seg = v.shape[1] // args.tensor_model_parallel_size
                    target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]

            elif "word_embeddings" in k or "output_layer" in k:
                seg = v.shape[0] // args.tensor_model_parallel_size
                target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
            else:
                target_v = v

            if "word_embeddings" in k:
                if pp_rank == 0:
                    model_split[k] = target_v
            elif "output_layer" in k or "final_layernorm" in k:
                if pp_rank == args.pipeline_model_parallel_size - 1:
                    model_split[k] = target_v
            else:
                model_split[k] = target_v
        save_state_dict(args, [model_split], checkpoint_name)

    print(f'megatron model is save to {args.save}')


def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

@torch.inference_mode()
def main():
    track_memory("MAIN_START")

    track_memory("BEFORE_INITIALIZE_MEGATRON")
    initialize_megatron(extra_args_provider=add_extra_args, allow_no_cuda=True)
    track_memory("AFTER_INITIALIZE_MEGATRON")

    args = get_args()
    track_memory("AFTER_GET_ARGS")

    if args.convert_checkpoint_from_megatron_to_transformers:
        track_memory_with_models("BEFORE_HF_CONFIG")
        config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        track_memory_with_models("AFTER_HF_CONFIG")

        track_memory_with_models("BEFORE_HF_MODEL_CREATION")
        # Keep HF model on meta device to avoid memory allocation
        with torch.device("meta"):
            hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=config.torch_dtype)
        track_memory_with_models("AFTER_HF_MODEL_CREATION_META", hf_model)

        # Note: Keeping model on meta device - no .to_empty() call to avoid 55GB allocation
        if ENABLE_DEBUG:
            print("🔍 HF model kept on meta device - zero memory allocation")

        track_memory_with_models("BEFORE_LOAD_MEGATRON_MODEL", hf_model)
        mg_model = load_megatron_model(args)
        track_memory_with_models("AFTER_LOAD_MEGATRON_MODEL", hf_model, mg_model)

        # AGGRESSIVE PRE-CONVERSION CLEANUP: Free all possible memory before conversion
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python to release memory back to OS
        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

        track_memory_with_models("BEFORE_INPLACE_CONV", hf_model, mg_model)
        convert_checkpoint_from_megatron_to_transformers_inplace(mg_model, hf_model, args)
        track_memory_with_models("AFTER_INPLACE_CONV", hf_model, mg_model)

        del mg_model
        gc.collect()
        track_memory_with_models("AFTER_DEL_MG", hf_model, None)

        hf_model.to("cpu")
        track_memory_with_models("BEFORE_HF_SAVE", hf_model, None)
        _assert_no_meta(hf_model)

        hf_model.save_pretrained(
            args.save,
            safe_serialization=True,
            max_shard_size=getattr(args, "hf_max_shard_size", "5GB"),
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
            tokenizer.save_pretrained(args.save)
        except Exception as exc:
            print(f"[warn] Tokenizer save skipped: {exc}")

        track_memory_with_models("AFTER_HF_SAVE", hf_model, None)
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        del hf_model
        gc.collect()
        save_mgmodel(mg_model, args)

    # Print complete memory usage summary
    print_memory_summary()


if __name__ == "__main__":
    main()

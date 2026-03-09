#!/usr/bin/env python3
# Copyright (c) 2026 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MAIN_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
MTP_LAYER_RE = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")
VISION_KEY_PREFIXES = (
    "model.visual.",
    "visual.",
    "model.vision_tower.",
    "model.vision_model.",
)


DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def parse_index(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_keep_list(raw: str, name: str) -> List[int]:
    if raw.strip() == "":
        raise ValueError(f"{name} cannot be empty")
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"{name} has negative index: {value}")
        out.append(value)
    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicates: {out}")
    return out


def parse_size(raw: str) -> int:
    s = raw.strip().lower()
    units = [
        ("kib", 2**10),
        ("mib", 2**20),
        ("gib", 2**30),
        ("tib", 2**40),
        ("kb", 10**3),
        ("mb", 10**6),
        ("gb", 10**9),
        ("tb", 10**12),
        ("b", 1),
    ]
    for unit, scale in units:
        if s.endswith(unit):
            num = float(s[: -len(unit)].strip())
            return int(num * scale)
    return int(s)


def _get_safetensor_header(shard_path: str) -> Dict:
    with open(shard_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), byteorder="little")
        return json.loads(f.read(header_len))


def _tensor_nbytes(dtype_name: str, shape: List[int]) -> int:
    if dtype_name not in DTYPE_NBYTES:
        raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")
    return int(math.prod(shape)) * DTYPE_NBYTES[dtype_name]


def _copy_non_weight_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        shutil.copy2(src, dst)


def _selected_weight_map(
    src_weight_map: Dict[str, str], keep_main: List[int], keep_mtp: List[int]
) -> Dict[str, Tuple[str, str]]:
    main_remap = {old: new for new, old in enumerate(keep_main)}
    mtp_remap = {old: new for new, old in enumerate(keep_mtp)}

    selected = {}
    for src_key, src_shard in src_weight_map.items():
        if src_key.startswith(VISION_KEY_PREFIXES):
            continue

        m = MAIN_LAYER_RE.match(src_key)
        if m is not None:
            old_layer = int(m.group(1))
            if old_layer not in main_remap:
                continue
            new_key = f"model.language_model.layers.{main_remap[old_layer]}.{m.group(2)}"
            selected[new_key] = (src_shard, src_key)
            continue

        m = MTP_LAYER_RE.match(src_key)
        if m is not None:
            old_layer = int(m.group(1))
            if old_layer not in mtp_remap:
                continue
            new_key = f"mtp.layers.{mtp_remap[old_layer]}.{m.group(2)}"
            selected[new_key] = (src_shard, src_key)
            continue

        selected[src_key] = (src_shard, src_key)

    if not selected:
        raise RuntimeError("No tensor selected after pruning; check keep layer arguments.")
    return selected


def _pack_keys_to_shards(
    selected: Dict[str, Tuple[str, str]], src_dir: str, max_shard_bytes: int
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    header_cache = {}
    nbytes = {}

    for dst_key, (src_shard, src_key) in selected.items():
        if src_shard not in header_cache:
            header_cache[src_shard] = _get_safetensor_header(os.path.join(src_dir, src_shard))
        meta = header_cache[src_shard].get(src_key)
        if meta is None:
            raise KeyError(f"Source key {src_key} not found in shard header {src_shard}")
        nbytes[dst_key] = _tensor_nbytes(meta["dtype"], meta["shape"])

    shard_to_keys = defaultdict(list)
    shard_id = 1
    current = 0
    for key in sorted(selected.keys()):
        size = nbytes[key]
        if current > 0 and current + size > max_shard_bytes:
            shard_id += 1
            current = 0
        shard_to_keys[shard_id].append(key)
        current += size
    return dict(shard_to_keys), nbytes


def _write_pruned_shards(
    selected: Dict[str, Tuple[str, str]], shard_to_keys: Dict[int, List[str]], src_dir: str, dst_dir: str
) -> Dict[str, str]:
    os.makedirs(dst_dir, exist_ok=True)
    total = len(shard_to_keys)
    weight_map_new = {}

    for shard_id in range(1, total + 1):
        keys = shard_to_keys[shard_id]
        src_group = defaultdict(list)
        for key in keys:
            src_shard, src_key = selected[key]
            src_group[src_shard].append((key, src_key))

        out = {}
        for src_shard, pairs in sorted(src_group.items()):
            shard_path = os.path.join(src_dir, src_shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for dst_key, src_key in pairs:
                    out[dst_key] = f.get_tensor(src_key)

        out_name = f"model-{shard_id:05d}-of-{total:05d}.safetensors"
        out_path = os.path.join(dst_dir, out_name)
        save_file(out, out_path)
        for key in keys:
            weight_map_new[key] = out_name
        print(f"Wrote {out_name} ({len(keys)} tensors)")
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return weight_map_new


def _update_config(config: Dict, keep_main: List[int], keep_mtp: List[int]) -> Dict:
    text = config.get("text_config", config)
    old_num_layers = int(text["num_hidden_layers"])
    old_layer_types = text.get("layer_types")

    if max(keep_main) >= old_num_layers:
        raise ValueError(
            f"keep-main-layers has out-of-range index; max index {max(keep_main)}, "
            f"num_hidden_layers {old_num_layers}"
        )
    text["num_hidden_layers"] = len(keep_main)
    if isinstance(old_layer_types, list):
        text["layer_types"] = [old_layer_types[i] for i in keep_main]

    old_mtp_layers = int(text.get("mtp_num_hidden_layers", 0))
    if old_mtp_layers > 0 and max(keep_mtp) >= old_mtp_layers:
        raise ValueError(
            f"keep-mtp-layers has out-of-range index; max index {max(keep_mtp)}, "
            f"mtp_num_hidden_layers {old_mtp_layers}"
        )
    text["mtp_num_hidden_layers"] = len(keep_mtp)

    if "text_config" in config:
        config["text_config"] = text
    if "num_hidden_layers" in config:
        config["num_hidden_layers"] = text["num_hidden_layers"]
    if "layer_types" in config and isinstance(config["layer_types"], list):
        config["layer_types"] = list(text.get("layer_types", config["layer_types"]))
    if "mtp_num_hidden_layers" in config:
        config["mtp_num_hidden_layers"] = text["mtp_num_hidden_layers"]
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Prune Qwen3.5 HF checkpoint by layer IDs and rebuild shard index."
    )
    parser.add_argument("--src-hf-dir", required=True)
    parser.add_argument("--dst-hf-dir", required=True)
    parser.add_argument(
        "--keep-main-layers",
        required=True,
        help="Comma-separated original model.language_model.layers indices to keep, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--keep-mtp-layers",
        default="0",
        help="Comma-separated original mtp.layers indices to keep, default: 0",
    )
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Max output shard size, e.g. 4GB, 2GiB, 5000000000",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_hf_dir)
    dst_dir = os.path.abspath(args.dst_hf_dir)
    keep_main = parse_keep_list(args.keep_main_layers, "keep-main-layers")
    keep_mtp = parse_keep_list(args.keep_mtp_layers, "keep-mtp-layers")
    max_shard_bytes = parse_size(args.max_shard_size)

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing source index: {index_path}")
    config_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing source config: {config_path}")

    print(f"Source HF: {src_dir}")
    print(f"Target HF: {dst_dir}")
    print(f"Keep main layers: {keep_main}")
    print(f"Keep mtp layers: {keep_mtp}")
    print(f"Max shard size: {max_shard_bytes} bytes")

    index = parse_index(index_path)
    src_weight_map = index["weight_map"]
    selected = _selected_weight_map(src_weight_map, keep_main, keep_mtp)
    shard_to_keys, selected_nbytes = _pack_keys_to_shards(selected, src_dir, max_shard_bytes)

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    _copy_non_weight_files(src_dir, dst_dir)
    new_weight_map = _write_pruned_shards(selected, shard_to_keys, src_dir, dst_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = _update_config(config, keep_main, keep_mtp)

    with open(os.path.join(dst_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    total_size = int(sum(selected_nbytes[k] for k in new_weight_map.keys()))
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Done. Kept {len(keep_main)} main layers and {len(keep_mtp)} mtp layers.")
    print(f"Output tensors: {len(new_weight_map)}, output shards: {len(shard_to_keys)}")


if __name__ == "__main__":
    main()

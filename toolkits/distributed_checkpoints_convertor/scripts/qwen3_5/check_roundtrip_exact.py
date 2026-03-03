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
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def _load_weight_map(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _load_header(shard_path):
    with open(shard_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), byteorder="little")
        return json.loads(f.read(header_len))


def _build_tensor_meta(model_dir, weight_map):
    keys_by_shard = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_shard[filename].append(key)

    meta = {}
    for filename, keys in keys_by_shard.items():
        shard_path = os.path.join(model_dir, filename)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard `{filename}` under {model_dir}")
        header = _load_header(shard_path)
        for key in keys:
            if key not in header:
                raise KeyError(f"Key `{key}` not found in shard header `{filename}`")
            tensor_meta = header[key]
            meta[key] = {
                "filename": filename,
                "dtype": tensor_meta["dtype"],
                "shape": tuple(tensor_meta["shape"]),
            }
    return meta


def _exclude_key(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            return True
    return False


def _tensor_max_abs_delta(a, b):
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        return (a.float() - b.float()).abs().max().item()
    if a.dtype in (torch.complex64, torch.complex128) or b.dtype in (torch.complex64, torch.complex128):
        return (a - b).abs().max().item()
    return float((a != b).sum().item())


def _load_tensor(model_dir, filename, key):
    with safe_open(os.path.join(model_dir, filename), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main():
    parser = argparse.ArgumentParser(
        description="Check exact HF roundtrip parity (keys, dtypes, values)."
    )
    parser.add_argument("--base-hf-dir", required=True, help="Original HF checkpoint directory.")
    parser.add_argument(
        "--roundtrip-hf-dir", required=True, help="Roundtripped HF checkpoint directory."
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=["model.visual."],
        help="Prefix to exclude from parity checks. Can be provided multiple times.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Max mismatches to print per check category.",
    )
    args = parser.parse_args()

    base_weight_map = _load_weight_map(args.base_hf_dir)
    roundtrip_weight_map = _load_weight_map(args.roundtrip_hf_dir)

    base_keys = {
        key for key in base_weight_map.keys() if not _exclude_key(key, args.exclude_prefix)
    }
    roundtrip_keys = {
        key for key in roundtrip_weight_map.keys() if not _exclude_key(key, args.exclude_prefix)
    }

    missing_keys = sorted(base_keys - roundtrip_keys)
    extra_keys = sorted(roundtrip_keys - base_keys)
    if missing_keys or extra_keys:
        print("FAILED: key-set mismatch")
        print(f"  missing_in_roundtrip={len(missing_keys)}")
        for key in missing_keys[: args.max_report]:
            print(f"    - {key}")
        print(f"  extra_in_roundtrip={len(extra_keys)}")
        for key in extra_keys[: args.max_report]:
            print(f"    + {key}")
        return 1

    shared_keys = sorted(base_keys)
    print(f"Comparing {len(shared_keys)} shared non-excluded keys...")

    base_meta = _build_tensor_meta(args.base_hf_dir, base_weight_map)
    roundtrip_meta = _build_tensor_meta(args.roundtrip_hf_dir, roundtrip_weight_map)

    dtype_mismatches = []
    shape_mismatches = []
    for key in shared_keys:
        b_meta = base_meta[key]
        r_meta = roundtrip_meta[key]
        if b_meta["dtype"] != r_meta["dtype"]:
            dtype_mismatches.append((key, b_meta["dtype"], r_meta["dtype"]))
        if b_meta["shape"] != r_meta["shape"]:
            shape_mismatches.append((key, b_meta["shape"], r_meta["shape"]))

    if dtype_mismatches or shape_mismatches:
        print("FAILED: metadata mismatch")
        print(f"  dtype_mismatches={len(dtype_mismatches)}")
        for key, b_dtype, r_dtype in dtype_mismatches[: args.max_report]:
            print(f"    - {key}: {b_dtype} != {r_dtype}")
        print(f"  shape_mismatches={len(shape_mismatches)}")
        for key, b_shape, r_shape in shape_mismatches[: args.max_report]:
            print(f"    - {key}: {b_shape} != {r_shape}")
        return 1

    value_mismatch_count = 0
    reported = 0
    worst_key = None
    worst_delta = -1.0
    for key in shared_keys:
        b_meta = base_meta[key]
        r_meta = roundtrip_meta[key]
        base_tensor = _load_tensor(args.base_hf_dir, b_meta["filename"], key)
        roundtrip_tensor = _load_tensor(args.roundtrip_hf_dir, r_meta["filename"], key)
        if torch.equal(base_tensor, roundtrip_tensor):
            continue

        value_mismatch_count += 1
        delta = _tensor_max_abs_delta(base_tensor, roundtrip_tensor)
        if delta > worst_delta:
            worst_delta = delta
            worst_key = key
        if reported < args.max_report:
            print(
                "VALUE_MISMATCH",
                key,
                f"dtype={base_tensor.dtype}",
                f"shape={tuple(base_tensor.shape)}",
                f"max_abs_delta={delta}",
            )
            reported += 1

    if value_mismatch_count > 0:
        print("FAILED: tensor value mismatch")
        print(f"  value_mismatch_count={value_mismatch_count}")
        print(f"  worst_key={worst_key}")
        print(f"  worst_max_abs_delta={worst_delta}")
        return 1

    print("PASSED: exact roundtrip parity on all shared non-excluded keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

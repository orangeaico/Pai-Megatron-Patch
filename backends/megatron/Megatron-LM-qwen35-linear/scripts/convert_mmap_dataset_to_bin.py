import os, json, numpy as np, sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIG ---
SRC_DIR = "/home/shared/qwen3-data-prep/extracted_data/"
OUT_DIR = os.path.join(SRC_DIR, "megatron_indexed")  # output dir
DEFAULT_SEQ_LEN = 8192
# --------------

path = lambda *p: os.path.abspath(os.path.join(*p))
src_dir = path(SRC_DIR)
out_dir = path(OUT_DIR)
os.makedirs(out_dir, exist_ok=True)

# metadata
meta = {}
mjson = path(src_dir, "metadata.json")
if os.path.exists(mjson):
    with open(mjson) as f: meta = json.load(f)
dtype = np.dtype(meta.get("dtype", "int32"))
seq_len = int(meta.get("seq_len", meta.get("block_size", DEFAULT_SEQ_LEN)))
total_blocks = meta.get("total_blocks", None)
shape = (total_blocks, seq_len) if total_blocks else tuple(meta.get("shape", []))

mmap_path = path(src_dir, "corpus.mmap")
if not os.path.exists(mmap_path):
    print(f"ERROR: {mmap_path} not found", file=sys.stderr); sys.exit(1)

mm = np.memmap(mmap_path, mode="r", dtype=dtype, shape=(tuple(shape) if shape else None))
two_d = (getattr(mm, "ndim", 1) == 2)

def get_builder(prefix_bin):
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    return IndexedDatasetBuilder(prefix_bin, dtype=np.int32), "core"

def write_split(split_name, idx_file):
    split_idx = path(src_dir, "indices", idx_file)
    if not os.path.exists(split_idx):
        print(f"WARNING: {split_idx} not found; skipping {split_name}", file=sys.stderr)
        return

    indices = np.load(split_idx)
    if indices.ndim > 1: indices = indices.squeeze()

    out_prefix = path(out_dir, f"{split_name}_text_document")
    bin_path = out_prefix + ".bin"
    idx_out  = out_prefix + ".idx"

    builder, variant = get_builder(bin_path)
    print(f"[{split_name}] Using builder: {variant}")
    print(f"[{split_name}] Writing prefix: {out_prefix}")

    # helper to emit one "document" (exactly one sequence)
    def add_one(seq_np: np.ndarray):
        # trim/pad to seq_len
        if seq_np.shape[0] != seq_len:
            seq_np = seq_np[:seq_len]
            if seq_np.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq_np.shape[0],), dtype=np.int32)
                seq_np = np.concatenate([seq_np, pad], axis=0)
        t = torch.from_numpy(seq_np.astype(np.int32, copy=False))
        # Prefer add_document if present; else add_item + end_document
        if hasattr(builder, "add_document"):
            builder.add_document(t, lengths=[t.numel()])
        else:
            builder.add_item(t)
            if hasattr(builder, "end_document"):
                builder.end_document()

    n = len(indices)
    for k, i in enumerate(indices):
        if two_d:
            arr = np.asarray(mm[int(i)], dtype=np.int32)
        else:
            start = int(i); end = start + seq_len
            arr = np.asarray(mm[start:end], dtype=np.int32)
        add_one(arr)
        if (k+1) % 1000 == 0 or (k+1) == n:
            print(f"[{split_name}] {k+1}/{n} items", file=sys.stderr)

    builder.finalize(idx_out)
    print(f"[{split_name}] Done:\n  {bin_path}\n  {idx_out}")

write_split("train", "train_idx.npy")
write_split("val",   "val_idx.npy")
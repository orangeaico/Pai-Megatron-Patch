#!/usr/bin/env bash
set -euo pipefail

# Always use your CUDA 12.9 ptxas (needed for sm_120)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Install Triton 3.3.x if not already
python - <<'PY' || pip install --no-deps --force-reinstall "triton==3.3.*"
import importlib.metadata as m
v = m.version("triton")
assert v.split(".")[0:2] >= ["3","3"]
print("triton ok:", v)
PY

# Install CCE if missing
python - <<'PY' || pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
import importlib
import cut_cross_entropy  # noqa
print("cce ok")
PY

# Patch CCE to add temperature scaling instead of softcapping 
python - <<'PY'
import ast
import os
from pathlib import Path
from typing import Dict, Optional

TARGET_FILES: Dict[Path, Dict[str, str]] = {
    Path("tl_utils.py"): {
        "tl_softcapping": "return v / softcap",
        "tl_softcapping_grad": "return dv / softcap",
    },
    Path("utils.py"): {
        "softcapping": "return logits / softcap",
    },
}

def locate_target(relative: Path) -> Path:
    try:
        import cut_cross_entropy
    except ImportError as exc:
        raise SystemExit(f"Failed to import cut_cross_entropy: {exc}") from exc
    module_path = Path(cut_cross_entropy.__file__).resolve()
    candidate = module_path.parent / relative
    if candidate.exists():
        return candidate
    raise SystemExit(f"Unable to locate {relative}")


def apply_replacements(target_path: Path, replacements: Dict[str, str]) -> bool:
    source_text = target_path.read_text()
    tree = ast.parse(source_text)
    lines = source_text.splitlines()
    updated = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in replacements:
            if not node.body:
                continue
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno - 1
            indent = " " * node.body[0].col_offset
            replacement_line = f"{indent}{replacements[node.name]}"
            if lines[start : end + 1] != [replacement_line]:
                lines[start : end + 1] = [replacement_line]
                updated = True

    if updated:
        target_path.write_text("\n".join(lines) + "\n")
    return updated


for relative_path, replacements in TARGET_FILES.items():
    target_file = locate_target(relative_path)
    if apply_replacements(target_file, replacements):
        print(f"Updated {target_file}")
    else:
        print(f"No updates required for {target_file}")
PY

# Install Transformers if missing
python - <<'PY' || pip install -U "transformers"
import importlib
import transformers  # noqa
try:
    from transformers import __version__ as v
except Exception:
    v = "unknown"
print("transformers ok:", v)
PY

# Clear old JIT caches (optional)
rm -rf ~/.cache/torch/inductor ~/.triton || true

unset PIP_CONSTRAINT
pip install --upgrade --no-cache-dir   "dill<0.3.9,>=0.3.0"   "datasets>=2.20.0"   "fsspec>=2024.6.1"   "huggingface_hub>=0.24.0"   "pyarrow>=12"
pip install jsonlines
pip install simpy
pip install --no-deps accelerate

if [ -n "${SETUP_FA3-}" ]; then
  # Install flash_attn_3
  pip install --no-index --no-deps \
    "https://huggingface.co/datasets/himanshu-livup/wheels/resolve/main/flash_attn_3-3.0.0b1+cu12-cp39-abi3-linux_x86_64.whl"

  # Install transformer_engine wheel
  pip uninstall -y transformer_engine transformer-engine || true
  PIP_CONSTRAINT=/dev/null pip install --no-index --no-deps \
    "https://huggingface.co/datasets/himanshu-livup/wheels/resolve/main/transformer_engine-2.9.0.dev0+4d14578-cp312-cp312-linux_x86_64.whl"

  PIP_CONSTRAINT=/dev/null pip install -U onnxscript==0.5.2 onnx_ir==0.1.9 ml_dtypes==0.5.3

  # Patch transformer_engine to use flash_attn_interface
  python - <<'PY'
import pathlib, re, sys

FILEPATH = "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/backends.py"

p = pathlib.Path(FILEPATH)
if not p.exists():
    print(f"[patch] ERROR: file not found: {FILEPATH}", file=sys.stderr)
    sys.exit(2)

src = p.read_text(encoding="utf-8")

# Replace only leading imports that reference flash_attn_3.flash_attn_interface
pat = re.compile(r'(?m)^(?P<i>\s*)(?P<kw>from|import)\s+flash_attn_3\.flash_attn_interface\b')
dst, n = pat.subn(r'\g<i>\g<kw> flash_attn_interface', src)

if n == 0:
    print(f"[patch] Nothing changed (already patched or different source): {FILEPATH}")
    sys.exit(0)

p.write_text(dst, encoding="utf-8")
print(f"[patch] Patched {n} line(s) in {FILEPATH}")
PY

python - <<'PY'
import transformer_engine as te
print("TE:", getattr(te,"__version__","n/a"))
import transformer_engine.pytorch as te_pt
print("TE PyTorch ok ->", te_pt.__file__)
PY

fi

pip install transformers==4.56.1
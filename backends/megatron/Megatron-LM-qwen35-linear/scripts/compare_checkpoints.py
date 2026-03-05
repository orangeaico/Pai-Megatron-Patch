# compare_pickles.py
import sys, pickle, torch

def load_dump(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    # ensure tensors
    for k, v in d.items():
        if not torch.is_tensor(v):
            d[k] = torch.as_tensor(v)
    return d

def main(a_path, b_path, atol=1e-6, rtol=1e-5, topk=10):
    A = load_dump(a_path)
    B = load_dump(b_path)

    keysA, keysB = set(A.keys()), set(B.keys())
    onlyA = sorted(keysA - keysB)
    onlyB = sorted(keysB - keysA)
    both  = sorted(keysA & keysB)

    # print ("keysA: ", keysA)

    print(f"A only: {len(onlyA)} | B only: {len(onlyB)} | both: {len(both)}")

    if onlyA[:topk]:
        print("  first few A-only:", onlyA[:topk])
    if onlyB[:topk]:
        print("  first few B-only:", onlyB[:topk])

    if len(both) == 0:
        print ("Error: Key mismatch between A and B, possibly comparing HF and megatron checkpoint pkls directly")
        sys.exit(1)
    diffs = []
    total_abs_diff = 0.0
    total_params = 0
    for k in both:
        ta, tb = A[k], B[k]
        if ta.shape != tb.shape:
            diffs.append((k, "shape_mismatch", ta.shape, tb.shape))
            continue
        # promote to common dtype for numeric comparison
        da = ta.detach()
        db = tb.detach()
        delta = (da - db).abs()
        max_abs = delta.max().item()
        # print(f"k: {k}, max_abs: {max_abs}")
        l2 = delta.pow(2).sum().sqrt().item()
        same = torch.allclose(da, db, atol=atol, rtol=rtol)
        diffs.append((k, same, max_abs, l2))
        
        # Add to total abs diff and param count
        total_abs_diff += delta.sum().item()
        total_params += ta.numel()
    
    bad = [(k, max_abs, l2) for (k, same, max_abs, l2) in diffs if same is not True]
    bad_shapes = [(k, s1, s2) for (k, tag, s1, s2) in diffs if tag == "shape_mismatch"]

    print(f"shape mismatches: {len(bad_shapes)}")
    for k, s1, s2 in bad_shapes[:topk]:
        print(f"  {k}: {s1} vs {s2}")

    print(f"value mismatches (not allclose): {len(bad)} (showing up to {topk})")
    for k, max_abs, l2 in sorted(bad, key=lambda x: (-x[1], -x[2]))[:topk]:
        print(f"  {k}: max_abs={max_abs:.6g} l2={l2:.6g}")
    
    # Print total statistics
    print(f"\nTotal statistics:")
    print(f"  Sum of absolute differences: {total_abs_diff:.6g}")
    print(f"  Total number of parameters: {total_params:,}")
    if total_params > 0:
        mean_abs_diff = total_abs_diff / total_params
        print(f"  Mean absolute difference: {mean_abs_diff:.6g}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_pickles.py <dumpA_rankX.pkl> <dumpB_rankX.pkl> [atol] [rtol]")
        sys.exit(1)
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    main(sys.argv[1], sys.argv[2], atol=atol, rtol=rtol)

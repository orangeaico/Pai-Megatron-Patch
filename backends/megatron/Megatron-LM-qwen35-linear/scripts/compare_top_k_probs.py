#!/usr/bin/env python3
import numpy as np

# ----------------------------
# Utilities (no SciPy needed)
# ----------------------------
def logsumexp_np(x: np.ndarray) -> float:
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def softmax_logprobs(x_log: np.ndarray) -> np.ndarray:
    """Return probabilities from log-probs x_log over the given set."""
    return np.exp(x_log - logsumexp_np(x_log))

def topk_indices_descending(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k values of x (descending by value)."""
    idx = np.argpartition(x, -k)[-k:]
    return idx[np.argsort(x[idx])[::-1]]

# ----------------------------
# Generate an LLM-like distro (spiky)
# ----------------------------
def generate_llm_like_logits_spiky(vocab_size=10_000, seed=42,
                                   alpha=0.8, noise_std=0.6,
                                   top_prob_target=0.96):
    """
    Create heavy-tailed logits and then adjust the top logit so that
    the final softmax has p_top ~= top_prob_target (> 95% as requested).
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(vocab_size)              # 0..V-1
    base = -alpha * np.log1p(ranks)            # monotone decreasing
    noise = rng.normal(0.0, noise_std, size=vocab_size)
    logits = base + noise

    # Find current top index under base logits
    top_idx = int(np.argmax(logits))

    # Compute logsumexp over *rest* (exclude top_idx)
    mask = np.ones(vocab_size, dtype=bool)
    mask[top_idx] = False
    lse_rest = logsumexp_np(logits[mask])
    # Set z_top so that p_top = exp(z_top) / (exp(z_top) + S_rest) = top_prob_target
    # z_top = log(S_rest) + log(p / (1-p))
    z_top = lse_rest + np.log(top_prob_target / (1.0 - top_prob_target))
    logits[top_idx] = z_top

    # Sanity check
    logZ = logsumexp_np(logits)
    p_top = np.exp(logits[top_idx] - logZ)
    assert p_top > 0.95, f"Top prob is {p_top:.4%}, expected > 95%"

    return logits

# ----------------------------
# Your renormalization (ported)
# ----------------------------
def user_renorm_probs_from_topk_logprobs(topk_logprobs: np.ndarray) -> np.ndarray:
    # Convert logprobs to probabilities and normalize via max-shift for stability
    max_logprob = np.max(topk_logprobs)
    exp_values = np.exp(topk_logprobs - max_logprob)
    normalized_probs = exp_values / np.sum(exp_values)
    return normalized_probs  # probabilities over the top-k set

# ----------------------------
# Main comparison
# ----------------------------
def main(vocab_size=10_000, top_k=50, seed=42):
    # 1) Build full distribution (spiky)
    logits_full = generate_llm_like_logits_spiky(
        vocab_size=vocab_size, seed=seed, top_prob_target=0.96
    )
    lse_full = logsumexp_np(logits_full)
    logprobs_full = logits_full - lse_full                 # full log-probs
    probs_full = np.exp(logprobs_full)                     # full probabilities (sum to 1)

    # 2) Top-50 by the FULL distribution
    topk_idx = topk_indices_descending(logprobs_full, top_k)
    topk_logprobs = logprobs_full[topk_idx]                # log p(i) for i in S
    topk_probs_true_full = probs_full[topk_idx]            # TRUE full probs for the top-50 (sum < 1 generally)
    mass_topk_true = float(np.sum(topk_probs_true_full))
    mass_tail_true = 1.0 - mass_topk_true

    # 3) Three top-50 probability constructions (over S)
    # A) topk_renorm_logprobs (conditionalize on S)
    lse_topk = logsumexp_np(topk_logprobs)
    renorm_logprobs_A = topk_logprobs - lse_topk
    probs_A = np.exp(renorm_logprobs_A)                    # probabilities over S

    # B) raw top-k -> softmax (equivalent to A up to fp error)
    probs_B = softmax_logprobs(topk_logprobs)

    # C) your renorm
    probs_C = user_renorm_probs_from_topk_logprobs(topk_logprobs)

    # Also compute "true conditional over top-k": p_true(i | i in S)
    probs_true_cond_topk = topk_probs_true_full / mass_topk_true

    # ----------------------------
    # Summaries
    # ----------------------------
    def summarize_diff(p, q, name_p, name_q):
        max_abs = float(np.max(np.abs(p - q)))
        l1 = float(np.sum(np.abs(p - q)))
        # symmetric KL guard
        eps = 1e-45
        p_ = np.clip(p, eps, 1.0)
        q_ = np.clip(q, eps, 1.0)
        kl_pq = float(np.sum(p_ * (np.log(p_) - np.log(q_))))
        kl_qp = float(np.sum(q_ * (np.log(q_) - np.log(p_))))
        print(f"[{name_p} vs {name_q}] max|Δ|={max_abs:.3e}  L1={l1:.3e}  KL(p||q)={kl_pq:.3e}  KL(q||p)={kl_qp:.3e}")

    top_id = int(topk_idx[0])
    print("=== Global properties ===")
    print(f"Vocab size: {vocab_size}, Top-K: {top_k}")
    print(f"Top token id: {top_id}")
    print(f"Top token true probability (full): {probs_full[top_id]:.6%}")
    print(f"Mass over top-{top_k} (true full): {mass_topk_true:.6%}")
    print(f"Mass over tail (V - top-{top_k}): {mass_tail_true:.6%}")

    print("\n=== Agreement of top-50 probabilities over S (expect ~identity) ===")
    summarize_diff(probs_true_cond_topk, probs_A, "True_cond(S)", "A:topk_renorm")
    summarize_diff(probs_true_cond_topk, probs_B, "True_cond(S)", "B:raw_topk→softmax")
    summarize_diff(probs_true_cond_topk, probs_C, "True_cond(S)", "C:user_renorm")
    summarize_diff(probs_A, probs_B, "A:topk_renorm", "B:raw_topk→softmax")
    summarize_diff(probs_A, probs_C, "A:topk_renorm", "C:user_renorm")

    # ----------------------------
    # Pretty table (all top-50)
    # ----------------------------
    header = (
        f"{'rank':>4} {'token_id':>8} "
        f"{'p_true_full':>14} {'p_true_cond(S)':>16} "
        f"{'p_A':>12} {'p_B':>12} {'p_C':>12}"
    )
    print("\nTop-50 comparison (shows TRUE full probs and 3 renormalized variants over S):")
    print(header)
    for r, t in enumerate(topk_idx):
        print(f"{r+1:>4} {int(t):>8} "
              f"{topk_probs_true_full[r]:>14.6e} {probs_true_cond_topk[r]:>16.6e} "
              f"{probs_A[r]:>12.6e} {probs_B[r]:>12.6e} {probs_C[r]:>12.6e}")

    # Checks
    assert np.allclose(probs_A, probs_B, rtol=1e-12, atol=1e-12)
    assert np.allclose(probs_A, probs_C, rtol=1e-12, atol=1e-12)
    assert np.allclose(probs_true_cond_topk, probs_A, rtol=1e-12, atol=1e-12)
    print("\nAll three top-50 constructions match the true conditional distribution over S within numerical precision.")

if __name__ == "__main__":
    main()

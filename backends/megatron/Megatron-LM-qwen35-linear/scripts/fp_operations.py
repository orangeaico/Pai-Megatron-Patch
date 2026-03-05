batch_size = 1
seq_length = 65000
num_layers = 48
hidden_size = 2048
ffn_hidden_size = 5472
moe_ffn_hidden_size = 768
num_experts_routed_to = 8
num_moe_layers = 48
num_dense_layers = 0
gated_linear_multiplier = 3 / 2
mtp_num_layers = 0
padded_vocab_size = 151936
shared_expert_ffn_hidden_size = 0
expansion_factor = 3 * 2 * 2
num_query_groups = 4
num_attention_heads = 32
kv_channels = 128
query_projection_size = kv_channels * num_attention_heads
query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

self_attn_term = (
                expansion_factor
                * num_layers
                * hidden_size
                * hidden_size
                * (
                    (
                        1
                        + (num_query_groups / num_attention_heads)
                        # # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                        + (seq_length / hidden_size / 2)
                    )
                    * query_projection_to_hidden_size_ratio
                )
            )

print (f"Self attention term: {self_attn_term/float(1024**4)}")

total_floating_point_operations = (
            batch_size
            * seq_length
            * (
                # MLP
                expansion_factor
                * num_layers
                * hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                    (ffn_hidden_size * gated_linear_multiplier)
                    * (num_dense_layers / num_layers)
                    # routed experts
                    + (moe_ffn_hidden_size * num_experts_routed_to * gated_linear_multiplier)
                    * (num_moe_layers / num_layers)
                    # Shared Experts.
                    + (shared_expert_ffn_hidden_size * gated_linear_multiplier)
                    * (num_moe_layers / num_layers)
                )
                # Self Attention
                + self_attn_term
                # MTP norms and proj
                + 3
                * 2
                * mtp_num_layers
                * (
                    # MTP eh norm + final nrom
                    3 * hidden_size
                    # MTH eh proj
                    + 2 * hidden_size * hidden_size
                )
                # Logit.
                + 3 * 2 * hidden_size * padded_vocab_size * (mtp_num_layers + 1)
            )
        )

print (f"Total floating point operations: {total_floating_point_operations/float(1024**4)}")
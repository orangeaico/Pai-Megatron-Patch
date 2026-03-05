# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.fusions.cce_loss import cce_per_token_loss
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import CudaGraphScope, ModelType
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_word_embeddings_state_dict,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    is_using_quantization_scales,
)

from megatron.core.fusions.distillation import traditional_distillation_loss, distillation_loss
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region

def _gather_full_sequence_across_context_parallel(
    logits: Tensor,
) -> Tensor:
    """All-gather context-parallel shards and restore original token order."""
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return logits
    cp_group = parallel_state.get_context_parallel_group()
    gathered: List[Tensor] = [torch.empty_like(logits) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, logits, group=cp_group)
    local_seq_len = logits.size(1)
    chunk_len = local_seq_len // 2
    if chunk_len == 0 or chunk_len * 2 != local_seq_len:
        raise RuntimeError("Context parallel sequence length is not balanced across ranks.")
    ordered_chunks: List[Optional[Tensor]] = [None] * (2 * cp_size)
    for rank, tensor_chunk in enumerate(gathered):
        first_half, second_half = torch.split(tensor_chunk, chunk_len, dim=1)
        ordered_chunks[rank] = first_half
        ordered_chunks[2 * cp_size - rank - 1] = second_half
    return torch.cat([chunk for chunk in ordered_chunks if chunk is not None], dim=1)

class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
        pg_collection (ProcessGroupCollection): Model communication process groups
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage
        self.disable_param_offloading = True

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=getattr(self.config, "yarn_rotary_scaling_factor"),
                original_max_position_embeddings=getattr(
                    self.config, "yarn_original_max_position_embeddings"
                ),
                beta_fast=getattr(self.config, "yarn_beta_fast"),
                beta_slow=getattr(self.config, "yarn_beta_slow"),
                mscale=getattr(self.config, "yarn_mscale"),
                mscale_all_dim=getattr(self.config, "yarn_mscale_all_dim"),
                correction_range_round_to_int=getattr(
                    self.config, "yarn_correction_range_round_to_int"
                ),
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = MultimodalRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
            self.mrope_section = self.config.mrope_section
            assert (
                self.mrope_section is not None
            ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage
            )

        # Output
        if self.post_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )
        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        # this is used to store combined cos/sin embeddings, exclusively for flash infer rope
        rotary_pos_cos_sin = None

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            use_flash_infer_fused_rope = (
                hasattr(inference_context, 'use_flashinfer_fused_rope')
                and inference_context.use_flashinfer_fused_rope
            )
            if in_inference_mode and (self.config.flash_decode or use_flash_infer_fused_rope):
                assert (
                    not self.config.flash_decode
                ) or inference_context.is_static_batching(), (
                    "Flash decode is only applicable to static batching."
                )
                # Flash decoding uses precomputed cos and sin for RoPE
                if self.config.flash_decode:
                    rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                        inference_context.max_sequence_length,
                        self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                    )
                elif use_flash_infer_fused_rope:
                    assert not self.mtp_process, "MTP not tested with flashinfer_fused_rope"
                    rotary_pos_cos_sin = self.rotary_pos_emb_cache.setdefault(
                        inference_context.max_sequence_length,
                        torch.cat(
                            self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                            -1,
                        ),
                    )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
        elif self.position_embedding_type == 'yarn':
            if self.training or not self.config.flash_decode:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb, _ = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
            else:
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "YarnRotaryEmbedding yet."
                )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(
                    position_ids,
                    self.mrope_section,
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            in_inference_mode
            and (
                (
                    self.config.cuda_graph_impl == "local"
                    and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
                )
                or self.config.flash_decode
            )
            and inference_context.is_static_batching()
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        else:
            sequence_len_offset = None

        if in_inference_mode:
            # Clear the outputs for padding tokens when using dynamic batching with
            # quantization scales to avoid corrupting amax calculations
            if inference_context.is_dynamic_batching() and is_using_quantization_scales(
                self.config
            ):
                decoder_input[inference_context.padding_slice] = 0.0

            # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
            # reference held by this caller function, enabling early garbage collection for
            # inference. Skip wrapping if decoder_input is logged after decoder completion.
            if not has_config_logger_enabled(self.config):
                decoder_input = WrappedTensor(decoder_input)

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        if rotary_pos_cos_sin is not None:
            # only in the case of flashinfer fused rope will we
            # return this extra tensor
            # this is for backwards compatibility with
            # legacy unit tests, which break if you
            # return a 6 tuple instead of 5.
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    def preprocess_for_fine_grained_offloading(self):
        """Preprocess for fine-grained activation offloading."""
        off_interface.init_chunk_handler(
            vp_size=self.config.virtual_pipeline_model_parallel_size,
            vp_stage=self.vp_stage,
            min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
        )
        if self.disable_param_offloading:
            for param in self.decoder.parameters():
                off_interface.mark_not_offloadable(param)
            if self.mtp_process:
                for param in self.mtp.parameters():
                    off_interface.mark_not_offloadable(param)
            if self.post_process:
                for param in self.output_layer.parameters():
                    off_interface.mark_not_offloadable(param)
            self.disable_param_offloading = False

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        teacher_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = (
            preproc_output[:5]
        )

        rotary_pos_cos_sin = preproc_output[5] if len(preproc_output) == 6 else None

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            teacher_data=teacher_data,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        teacher_data=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        cce_fast_path = (
            labels is not None
            and self.config.use_linear_cross_entropy
            and parallel_state.is_pipeline_last_stage()
        )
        cce_sequence_parallel_override = False
        cce_inputs_prepared = False
        cce_classifier_weight = None
        cce_vocab_size = None
        cce_probe_done = False

        if self.mtp_process:
            if cce_fast_path and self.config.sequence_parallel:
                tp_group = self.pg_collection.tp if self.pg_collection is not None else None
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=tp_group
                )
                cce_inputs_prepared = True
                if self.output_layer.sequence_parallel:
                    self.output_layer.sequence_parallel = False
                    cce_sequence_parallel_override = True
                if self.config.debug_cce_loss and parallel_state.get_tensor_model_parallel_rank() == 0:
                    print(
                        f"[CCE debug] gathered hidden_states {hidden_states.shape}, labels {labels.shape}"
                    )

            if cce_fast_path:
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()
                else:
                    output_weight = None

                cce_classifier_weight = self.output_layer.weight if output_weight is None else output_weight
                cce_vocab_size = getattr(self, "padded_vocab_size", None) or \
                    getattr(self, "vocab_size", cce_classifier_weight.size(0))

                # If embeddings and output weights are UNTIED, Megatron's DDP expects the
                # output layer module to be invoked so its param-gather hooks run.
                # Trigger hooks with a *minimal*, zero-grad probe that does NOT stash activations.
                if not self.share_embeddings_and_output_weights:
                    B = hidden_states.size(1)
                    H = hidden_states.size(2)
                    probe = torch.empty(
                        (1, B, H),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    _old_eab = getattr(self.output_layer, "embedding_activation_buffer", None)
                    _old_gob = getattr(self.output_layer, "grad_output_buffer", None)
                    try:
                        if hasattr(self.output_layer, "embedding_activation_buffer"):
                            self.output_layer.embedding_activation_buffer = None
                        if hasattr(self.output_layer, "grad_output_buffer"):
                            self.output_layer.grad_output_buffer = None
                        with torch.no_grad():
                            _probe_logits, _ = self.output_layer(
                                probe, runtime_gather_output=False
                            )
                    finally:
                        if hasattr(self.output_layer, "embedding_activation_buffer"):
                            self.output_layer.embedding_activation_buffer = _old_eab
                        if hasattr(self.output_layer, "grad_output_buffer"):
                            self.output_layer.grad_output_buffer = _old_gob
                    cce_probe_done = True

            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_num_layers):
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, num_tokens = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                if cce_fast_path:
                    mtp_loss = cce_per_token_loss(
                        embeddings=hidden_states_list[mtp_layer_number + 1],
                        classifier_weight=cce_classifier_weight,
                        labels=mtp_labels,
                        vocab_size=cce_vocab_size,
                        impl=self.config.linear_ce_impl,
                        reduction=self.config.linear_ce_reduction,
                        shift=self.config.linear_ce_shift,
                        ignore_index=self.config.linear_ce_ignore_index,
                    )
                else:
                    # output
                    mtp_logits, _ = self.output_layer(
                        hidden_states_list[mtp_layer_number + 1],
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )
                    mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    # TODO(shifangx): remove the use of parallel_state here
                    # after moving loss logging to loss_func in pretrain_gpt.py
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )

        # -------- Cut Cross-Entropy fast path (no logits materialization) -------
        # Only compute loss on the LAST PP stage; others take the default path.
        if cce_fast_path:
            if self.config.distillation_with_traditional and self.config.distillation_loss:
                logits, _ = self.output_layer(
                    hidden_states,
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                logits = gather_from_tensor_model_parallel_region(
                    logits, group=self.pg_collection.tp
                )
                logits = logits.transpose(0, 1).contiguous()  # [B, T, V]
                logits = _gather_full_sequence_across_context_parallel(
                    logits
                )
                labels_cp = _gather_full_sequence_across_context_parallel(
                    labels
                )
                traditional_kl_loss = traditional_distillation_loss(
                    logits,
                    teacher_data,
                    labels_cp,
                    T=self.config.distillation_temperature,
                    ignore_index=self.config.linear_ce_ignore_index,
                )
                    
            sequence_parallel_override = cce_sequence_parallel_override
            if self.config.sequence_parallel and not cce_inputs_prepared:
                tp_group = self.pg_collection.tp if self.pg_collection is not None else None
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=tp_group
                )
                if self.output_layer.sequence_parallel:
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True
                if self.config.debug_cce_loss and parallel_state.get_tensor_model_parallel_rank() == 0:
                    print(
                        f"[CCE debug] gathered hidden_states {hidden_states.shape}, labels {labels.shape}"
                    )

            # Choose classifier weights (tied vs. untied).
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()
            else:
                output_weight = None
            
            if cce_classifier_weight is not None:
                classifier_weight = cce_classifier_weight
                vocab_size = cce_vocab_size
            elif output_weight is None:
                classifier_weight = self.output_layer.weight
                vocab_size = getattr(self, "padded_vocab_size", None) or \
                             getattr(self, "vocab_size", classifier_weight.size(0))
            else:
                classifier_weight = output_weight
                vocab_size = getattr(self, "padded_vocab_size", None) or \
                             getattr(self, "vocab_size", classifier_weight.size(0))

            if self.config.debug_cce_loss:
            # Reference loss implementation
                reference_logits, _ = self.output_layer(
                    hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
                )
                reference_loss = self.compute_language_model_loss(labels, reference_logits)
                print (f"Reference cross entropy loss: {reference_loss}")

            # If embeddings and output weights are UNTIED, Megatron's DDP expects the
            # output layer module to be invoked so its param-gather hooks run.
            # Trigger hooks with a *minimal*, zero-grad probe that does NOT stash activations.
            if not self.share_embeddings_and_output_weights and not cce_probe_done:
                # Build a fresh 1-token probe in [S, B, H] layout to avoid aliasing hidden_states.
                B = hidden_states.size(1)
                H = hidden_states.size(2)
                probe = torch.empty(
                    (1, B, H),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                # Temporarily disable any activation/grad buffers on the output layer
                # to avoid storing probe activations (important for defer_embedding_wgrad_compute).
                _old_eab = getattr(self.output_layer, "embedding_activation_buffer", None)
                _old_gob = getattr(self.output_layer, "grad_output_buffer", None)
                try:
                    if hasattr(self.output_layer, "embedding_activation_buffer"):
                        self.output_layer.embedding_activation_buffer = None
                    if hasattr(self.output_layer, "grad_output_buffer"):
                        self.output_layer.grad_output_buffer = None
                    with torch.no_grad():
                        # Call without overriding 'weight' so module uses its own params.
                        # runtime_gather_output=False prevents TP gather for the probe.
                        _probe_logits, _ = self.output_layer(
                            probe, runtime_gather_output=False
                        )
                finally:
                    if hasattr(self.output_layer, "embedding_activation_buffer"):
                        self.output_layer.embedding_activation_buffer = _old_eab
                    if hasattr(self.output_layer, "grad_output_buffer"):
                        self.output_layer.grad_output_buffer = _old_gob

            if self.config.debug_cce_loss:
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                world_rank = torch.distributed.get_rank()
                print(
                    f"[CCE debug] rank={world_rank} tp_rank={tp_rank} vocab_size={vocab_size}"
                )

            token_losses = cce_per_token_loss(
                embeddings=hidden_states,             # [B, T, H]
                classifier_weight=classifier_weight,  # [V, H] (or [V_local, H])
                labels=labels,                        # [B, T]
                vocab_size=vocab_size,
                impl=self.config.linear_ce_impl,
                reduction=self.config.linear_ce_reduction,
                shift=self.config.linear_ce_shift,
                ignore_index=self.config.linear_ce_ignore_index,
            )

            if self.config.distillation_loss:
                kl_loss = distillation_loss(
                embeddings=hidden_states,  # [B, T, H]
                classifier_weight=classifier_weight,  # [V, H] (or [V_local, H])
                labels=labels,  # [B, T]
                vocab_size=vocab_size,
                impl=self.config.linear_ce_impl,
                reduction=self.config.linear_ce_reduction,
                shift=self.config.linear_ce_shift,
                ignore_index=self.config.linear_ce_ignore_index,
                debug=self.config.debug_distillation,
                temperature=self.config.distillation_temperature,
                teacher_data=teacher_data,
            )

            if self.config.debug_cce_loss:
                if parallel_state.get_tensor_model_parallel_world_size() > 1:
                    tp_rank_local = parallel_state.get_tensor_model_parallel_rank()
                else:
                    tp_rank_local = 0
                # Debug: Print CCE loss and compare with reference loss
                print("\n=== CCE vs Reference Loss Comparison ===")
                print(f"[Rank {tp_rank_local}] Reference loss shape: {reference_loss.shape}, CCE loss shape: {token_losses.shape}")
                print(f"[Rank {tp_rank_local}] Loss mask shape: {loss_mask.shape if loss_mask is not None else 'None'}")

                print(f"[Rank {tp_rank_local}] Reference loss stats - Min: {reference_loss.min().item()}, Max: {reference_loss.max().item()}, Mean: {reference_loss.mean().item()}")
                print(f"[Rank {tp_rank_local}] CCE loss stats - Min: {token_losses.min().item()}, Max: {token_losses.max().item()}, Mean: {token_losses.mean().item()}")
                
                # Check first 10 values
                print(f"\n[Rank {tp_rank_local}] First 10 reference losses: {reference_loss.flatten()[:10].tolist()}")
                print(f"[Rank {tp_rank_local}] First 10 CCE losses: {token_losses.flatten()[:10].tolist()}")
                print(f"[Rank {tp_rank_local}] First 10 labels: {labels.flatten()[:10].tolist()}")

                print(f"\n[Rank {tp_rank_local}] Last 10 reference losses: {reference_loss.flatten()[-10:].tolist()}")
                print(f"[Rank {tp_rank_local}] Last 10 CCE losses: {token_losses.flatten()[-10:].tolist()}")
                print(f"[Rank {tp_rank_local}] Last 10 labels: {labels.flatten()[-10:].tolist()}")
                
                # Check for shape mismatch - CCE might be returning [B, T] while reference is [T, B]
                if reference_loss.shape != token_losses.shape:
                    print(f"\n[Rank {tp_rank_local}] WARNING: Shape mismatch! Attempting to compare after transpose...")
                    if token_losses.shape == (reference_loss.shape[1], reference_loss.shape[0]):
                        token_losses_transposed = token_losses.transpose(0, 1)
                        diff = (reference_loss - token_losses_transposed).abs()
                        print(f"[Rank {tp_rank_local}] Difference after transpose - Min: {diff.min().item():.6f}, Max: {diff.max().item():.6f}, Mean: {diff.mean().item():.6f}")
                else:
                    # Direct comparison
                    diff = (reference_loss - token_losses).abs()
                    print(f"\n[Rank {tp_rank_local}] Difference between reference and CCE - Min: {diff.min().item()}, Max: {diff.max().item()}, Mean: {diff.mean().item()}")
                
                # Check if CCE is returning negative values
                neg_count = (token_losses < 0).sum().item()
                if neg_count > 0:
                    print(f"\n[Rank {tp_rank_local}] WARNING: CCE returned {neg_count} negative values!")
                    print(f"[Rank {tp_rank_local}] Negative values: {token_losses[token_losses < 0].flatten()[:10].tolist()}")
                    
                    # Find indices where CCE returned negative values
                    neg_mask = token_losses < 0
                    neg_indices = torch.where(neg_mask)
                    
                    # Print reference loss and labels at those indices
                    print(f"\n[Rank {tp_rank_local}] Analyzing negative CCE values:")
                    num_to_analyze = min(10, len(neg_indices[0])) 
                    for i in range(num_to_analyze):
                        batch_idx = neg_indices[0][i].item()
                        seq_idx = neg_indices[1][i].item()
                        cce_val = token_losses[batch_idx, seq_idx].item()
                        ref_val = reference_loss[batch_idx, seq_idx].item() if reference_loss.shape == token_losses.shape else reference_loss[seq_idx, batch_idx].item()
                        label_val = labels[batch_idx, seq_idx].item()
                        
                        # Get loss_mask value at this position
                        if loss_mask is not None:
                            mask_val = loss_mask[batch_idx, seq_idx].item() if loss_mask.shape == labels.shape else loss_mask[seq_idx, batch_idx].item()
                        else:
                            mask_val = "No mask"
                        
                        # Check if label is in local vocab range
                        if parallel_state.get_tensor_model_parallel_world_size() > 1:
                            from megatron.core.tensor_parallel.utils import VocabUtility
                            tp_rank_local = parallel_state.get_tensor_model_parallel_rank()
                            tp_world_local = parallel_state.get_tensor_model_parallel_world_size()
                            start, end = VocabUtility.vocab_range_from_global_vocab_size(vocab_size, tp_rank_local, tp_world_local)
                            in_range = start <= label_val < end if label_val >= 0 else False
                            print(f"  Index [{batch_idx}, {seq_idx}]: CCE={cce_val}, Reference={ref_val}, Label={label_val}, Mask={mask_val}, InLocalVocab={in_range}")
                        else:
                            print(f"  Index [{batch_idx}, {seq_idx}]: CCE={cce_val}, Reference={ref_val}, Label={label_val}, Mask={mask_val}")
                    
                    # Analyze which labels tend to produce negative values
                    print(f"\n[Rank {tp_rank_local}] Analyzing labels that produce negative CCE values:")
                    neg_labels = []
                    for i in range(len(neg_indices[0])):
                        batch_idx = neg_indices[0][i].item()
                        seq_idx = neg_indices[1][i].item()
                        neg_labels.append(labels[batch_idx, seq_idx].item())
                    
                    from collections import Counter
                    label_counts = Counter(neg_labels)
                    print(f"[Rank {tp_rank_local}] Top 10 labels with negative CCE values: {label_counts.most_common(10)}")
                        
                else:
                    print(f"\n[Rank {tp_rank_local}] CCE returned no negative values!")
                
                # Check if reference loss is returning negative values
                neg_count = (reference_loss < 0).sum().item()
                if neg_count > 0:
                    print(f"\n[Rank {tp_rank_local}] WARNING: Reference loss returned {neg_count} negative values!")
                    print(f"[Rank {tp_rank_local}]    Negative values: {reference_loss[reference_loss < 0].flatten()[:10].tolist()}")
                else:
                    print(f"\n[Rank {tp_rank_local}] Reference loss returned no negative values!")

            if self.config.trsft:
                target_log_probs = -token_losses
                target_probs = torch.exp(target_log_probs)
                if self.config.trsft_alpha > 0:
                    gradient_scale = torch.clamp(target_probs / self.config.trsft_alpha, max=1.0)
                else:
                    gradient_scale = torch.ones_like(target_probs)
                gradient_scale = gradient_scale.detach()
                token_losses *= gradient_scale

            if self.config.return_logits_when_using_cce:
                # Compute logits after the fact only if explicitly requested.
                logits, _ = self.output_layer(
                    hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
                )
                if sequence_parallel_override:
                    self.output_layer.sequence_parallel = True
                return token_losses, logits

            # Default: return per-token losses for Megatron's loss masking/reduction.
            if sequence_parallel_override:
                self.output_layer.sequence_parallel = True

            if self.config.distillation_loss:
                alpha = self.config.distillation_loss_alpha
                distill_loss = alpha * token_losses + (1 - alpha) * kl_loss
                if self.config.debug_distillation:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    cp_rank = parallel_state.get_context_parallel_rank()
                    world_rank = torch.distributed.get_rank()
                    print(f"[Distillation debug] rank={world_rank} tp_rank={tp_rank} cp_rank={cp_rank} CCE Loss: {token_losses.sum().item()}, KL Loss: {kl_loss.sum().item()}, Distill Loss: {distill_loss.sum().item()}")
                    print(
                        f"[Distillation debug] rank={world_rank} tp_rank={tp_rank} cp_rank={cp_rank} distill_loss stats - Min: {distill_loss.min().item()}, Max: {distill_loss.max().item()}, Mean: {distill_loss.mean().item()} Sum: {distill_loss.sum().item()}"
                    )
                return distill_loss

            return token_losses

        sequence_parallel_override = False

        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [B, 1, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        if self.config.trsft:
            target_log_probs = -loss
            target_probs = torch.exp(target_log_probs)
            if self.config.trsft_alpha > 0:
                gradient_scale = torch.clamp(target_probs / self.config.trsft_alpha, max=1.0)
            else:
                gradient_scale = torch.ones_like(target_probs)
            gradient_scale = gradient_scale.detach()
            loss *= gradient_scale

        return loss

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ):
        """Builds a computation schedule plan for the model.

        This function creates a schedule plan for a model chunk, including
        preprocessing, transformer layers, and postprocessing.
        The schedule plan is used to optimize computation and memory usage
        in distributed environments.

        Args:
            input_ids (Tensor): Input token IDs.
            position_ids (Tensor): Position IDs.
            attention_mask (Tensor): Attention mask.
            decoder_input (Tensor, optional): Decoder input tensor. Defaults to None.
            labels (Tensor, optional): Labels for loss computation. Defaults to None.
            inference_context (BaseInferenceContext, optional):
                Inference context. Defaults to None.
            packed_seq_params (PackedSeqParams, optional):
                Parameters for packed sequences. Defaults to None.
            extra_block_kwargs (dict, optional):
                Additional keyword arguments for blocks. Defaults to None.
            runtime_gather_output (Optional[bool], optional):
                Whether to gather output at runtime. Defaults to None.
            inference_params (InferenceParams, optional):
                Parameters for inference. Defaults to None.
            loss_mask (Optional[Tensor], optional): Loss mask. Defaults to None.

        Returns:
            TransformerModelChunkSchedulePlan: The model chunk schedule plan.
        """

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        from ..common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

        return TransformerModelChunkSchedulePlan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
        )

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.

        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        # Multi-Token Prediction (MTP) need embedding layer in mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Now MTP loss is computed in post processing stage, so the output_layer is not needed.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(
                sharded_state_dict,
                emb_weight,
                emb_weight_key,
                tp_group=self.tp_group,
                dp_cp_group=metadata['dp_cp_group'],
            )

        return sharded_state_dict

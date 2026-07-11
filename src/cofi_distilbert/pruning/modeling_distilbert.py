"""
Clean-room CoFi-style reimplementation of DistilBERT, ported from
``models/modeling_bert.py`` in the official CoFiPruning repository.

Weight names in every class below match ``transformers``' stock
``distilbert-base-uncased`` exactly (``q_lin``, ``k_lin``, ``v_lin``,
``out_lin``, ``sa_layer_norm``, ``ffn.lin1``, ``ffn.lin2``,
``output_layer_norm``, ``pre_classifier``, ``classifier``), so a plain
``state_dict()`` from ``AutoModelForSequenceClassification`` loads into
these classes with ``strict=True`` (before any pruning) and a pruned
checkpoint saved from these classes loads back into stock
``DistilBertForSequenceClassification`` for any layer that ended up
untouched.

Mapping from the official BERT port to DistilBERT:

    CoFiBertEmbeddings        -> CoFiDistilBertEmbeddings   (no token_type embeddings)
    CoFiBertSelfAttention     -> CoFiMultiHeadSelfAttention  (q_lin/k_lin/v_lin)
    CoFiBertSelfOutput        -> folded into CoFiMultiHeadSelfAttention.out_lin +
                                  CoFiTransformerBlock.sa_layer_norm (DistilBERT does not
                                  split "self output projection" into its own submodule
                                  the way BERT's BertSelfOutput does)
    CoFiBertOutput             -> CoFiFFN + CoFiTransformerBlock.output_layer_norm
    CoFiBertLayer              -> CoFiTransformerBlock
    CoFiBertEncoder             -> CoFiTransformer
    CoFiBertModel               -> CoFiDistilBertModel
    CoFiBertForSequenceClassification -> CoFiDistilBertForSequenceClassification

Every forward method below accepts the same five z-tensors the official
code threads through BERT: ``head_z``, ``head_layer_z``, ``intermediate_z``,
``mlp_z``, ``hidden_z``. During CoFi training these are soft (fractional,
differentiable) masks sampled from ``DistilBertL0Module``; at hard-pruning
time they are removed entirely because ``cofi_utils.prune_model_with_z``
physically resizes the underlying ``nn.Linear``/``nn.LayerNorm`` modules
instead, exactly mirroring the official two-stage design (soft mask during
training -> physical resize once for the final checkpoint).
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Set, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
    DistilBertModel,
    DistilBertPreTrainedModel,
)

logger = logging.getLogger(__name__)


class CoFiLayerNorm(torch.nn.LayerNorm):
    """
    Identical to the official ``CoFiLayerNorm``: a LayerNorm that can
    optionally compress its input/weight/bias down to only the hidden
    dimensions that survive ``hidden_z`` before normalizing, so soft
    hidden-dimension masking during training behaves like the eventual
    physical pruning of those dimensions.
    """

    def __init__(self, normalized_shape, eps: float = 1e-12, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[..., remaining_index] = normed_input
        else:
            output = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output


class CoFiDistilBertEmbeddings(nn.Module):
    """
    Port of ``CoFiBertEmbeddings``, minus token-type embeddings (DistilBERT
    has none). Word + position embeddings, optionally masked by
    ``hidden_z`` before and after the LayerNorm, exactly like the official
    BERT embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = CoFiLayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids=None, input_embeds=None, hidden_z=None):
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)

        seq_length = input_embeds.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeddings
        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        embeddings = self.LayerNorm(embeddings, hidden_z)
        embeddings = self.dropout(embeddings)
        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        return embeddings


class CoFiMultiHeadSelfAttention(nn.Module):
    """
    Port of ``CoFiBertSelfAttention`` + ``CoFiBertSelfOutput`` combined,
    because DistilBERT keeps the output projection (``out_lin``) inside the
    attention module itself rather than in a separate "SelfOutput"
    submodule the way BERT does. ``head_z`` soft-masks each head's
    contribution to the context vector (mirrors
    ``context_layer *= head_z`` in the official code); ``head_layer_z``
    soft-masks the whole layer's attention output post-projection (mirrors
    ``CoFiBertSelfOutput``'s ``hidden_states.mul(head_layer_z)``).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.q_lin = nn.Linear(config.dim, config.dim)
        self.k_lin = nn.Linear(config.dim, config.dim)
        self.v_lin = nn.Linear(config.dim, config.dim)
        self.out_lin = nn.Linear(config.dim, config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        """
        Physical head pruning. Extends the stock
        ``MultiHeadSelfAttention.prune_heads`` (which assumes at least one
        head always survives) with the official CoFi behaviour of setting
        every linear to ``None`` when *all* heads in a layer are pruned,
        matching ``CoFiBertAttention.prune_heads``'s handling of a fully
        zeroed-out ``head_z`` layer.
        """
        len_heads = len(heads)
        if len_heads == 0:
            return
        if self.q_lin is None:
            # Already fully pruned in a previous call.
            return

        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.attention_head_size, self.pruned_heads)

        if len(index) == 0:
            self.q_lin = None
            self.k_lin = None
            self.v_lin = None
            self.out_lin = None
        else:
            self.q_lin = prune_linear_layer(self.q_lin, index)
            self.k_lin = prune_linear_layer(self.k_lin, index)
            self.v_lin = prune_linear_layer(self.v_lin, index)
            self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)

        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if self.q_lin is None:
            # The whole layer was physically pruned away.
            return (None, None) if output_attentions else (None,)

        bs, q_length, _ = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads

        def shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))

        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask_reshp = (bs, 1, 1, k_length)
        attn_mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        if head_z is not None:
            # head_z: (n_heads,) or broadcastable to (1, n_heads, 1, 1) -- soft-masks
            # each head's contribution to the context vector before the output
            # projection, exactly like the official CoFiBertSelfAttention.
            context = context * head_z.view(1, -1, 1, 1)

        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if head_layer_z is not None:
            context = context.mul(head_layer_z)

        if output_attentions:
            return (context, weights)
        return (context,)


class CoFiFFN(nn.Module):
    """
    Port of the FFN half of ``CoFiBertOutput``. ``intermediate_z``
    soft-masks individual FFN neurons (post-activation, pre ``lin2``,
    mirroring ``intermediate_output.mul(intermediate_z)``); ``mlp_z``
    soft-masks the whole FFN sublayer's contribution.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0)
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(config.dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.dim)
        self.activation = nn.GELU() if config.activation == "gelu" else nn.ReLU()

    def forward(
        self,
        input: torch.Tensor,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.lin1 is None:
            return None
        if self.chunk_size_feed_forward == 0:
            # Note: intermediate_z/mlp_z are passed as closure arguments rather than
            # stored as instance attributes on `self`. Stashing graph-connected
            # (non-leaf) tensors as plain module attributes breaks `copy.deepcopy`,
            # which `cofi_utils.generate_compact_model` relies on.
            return self.ff_chunk(input, intermediate_z, mlp_z)

        def chunk_fn(chunk: torch.Tensor) -> torch.Tensor:
            return self.ff_chunk(chunk, intermediate_z, mlp_z)

        return apply_chunking_to_forward(chunk_fn, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(
        self,
        input: torch.Tensor,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        if intermediate_z is not None:
            x = x.mul(intermediate_z)
        x = self.lin2(x)
        if mlp_z is not None:
            x = x.mul(mlp_z)
        x = self.dropout(x)
        return x


class CoFiTransformerBlock(nn.Module):
    """
    Port of ``CoFiBertLayer``. Wires ``head_z``/``head_layer_z`` through
    attention and ``intermediate_z``/``mlp_z`` through the FFN, with the
    same "all-zero shortcut" the official code uses at inference time: if
    an entire sublayer's contribution rounds to zero, its residual
    connection is skipped instead of adding a zero tensor and paying for a
    LayerNorm over it.
    """

    def __init__(self, config):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = CoFiMultiHeadSelfAttention(config)
        self.sa_layer_norm = CoFiLayerNorm(config.dim, eps=1e-12)
        self.ffn = CoFiFFN(config)
        self.output_layer_norm = CoFiLayerNorm(config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_attn_outputs = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
        )
        sa_output = self_attn_outputs[0]

        if sa_output is None:
            # Entire attention sublayer physically pruned away: passthrough.
            sa_output = x
        elif not inference and sa_output.sum().eq(0).item():
            sa_output = sa_output + x
        else:
            if hidden_z is not None:
                sa_output = sa_output.mul(hidden_z)
            sa_output = self.sa_layer_norm(sa_output + x, hidden_z)
            if hidden_z is not None:
                sa_output = sa_output.mul(hidden_z)

        if self.ffn.lin1 is None:
            ffn_output = sa_output
        else:
            raw_ffn_output = self.ffn(sa_output, intermediate_z=intermediate_z, mlp_z=mlp_z)
            if not inference and raw_ffn_output.sum().eq(0).item():
                ffn_output = raw_ffn_output + sa_output
            else:
                if hidden_z is not None:
                    raw_ffn_output = raw_ffn_output.mul(hidden_z)
                ffn_output = self.output_layer_norm(raw_ffn_output + sa_output, hidden_z)
                if hidden_z is not None:
                    ffn_output = ffn_output.mul(hidden_z)

        outputs = (ffn_output,)
        if output_attentions:
            outputs = (self_attn_outputs[1],) + outputs
        return outputs


class CoFiTransformer(nn.Module):
    """Port of ``CoFiBertEncoder``: stacks ``CoFiTransformerBlock``s and threads per-layer zs through."""

    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([CoFiTransformerBlock(config) for _ in range(config.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        inference: bool = False,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                hidden_state,
                attn_mask,
                output_attentions=output_attentions,
                head_z=head_z[i] if head_z is not None else None,
                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                mlp_z=mlp_z[i] if mlp_z is not None else None,
                hidden_z=hidden_z,
                inference=inference,
            )
            hidden_state = layer_outputs[-1]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[0],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)


class CoFiDistilBertModel(DistilBertPreTrainedModel):
    """Port of ``CoFiBertModel``: embeddings + transformer, both z-aware."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = CoFiDistilBertEmbeddings(config)
        self.transformer = CoFiTransformer(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        inference: bool = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        embedding_output = self.embeddings(input_ids=input_ids, input_embeds=inputs_embeds, hidden_z=hidden_z)

        return self.transformer(
            x=embedding_output,
            attn_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
            inference=inference,
        )


class CoFiDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    """
    Port of ``CoFiBertForSequenceClassification``. Wires ``pre_classifier``
    and ``classifier`` through ``hidden_z`` (their input dimension shrinks
    whenever hidden dimensions are pruned), and optionally exposes a
    ``layer_transformation`` projection for layer-wise distillation, exactly
    mirroring the official ``do_layer_distill`` flag.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = CoFiDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)
        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.dim, config.dim)
        else:
            self.layer_transformation = None

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Loads a *dense* (unpruned) checkpoint into the CoFi classes, e.g. a
        fine-tuned ``distilbert-base-uncased`` teacher, so it can be used
        as the starting point for CoFi training. For loading an already
        physically-pruned compact checkpoint (where layer shapes vary),
        use ``cofi_utils.load_compact_model`` instead -- that path infers
        each layer's shape from the checkpoint itself before construction.
        """
        if os.path.isdir(pretrained_model_name_or_path):
            weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                weights = torch.load(weights_path, map_location="cpu")
            else:
                # Fall back to safetensors via the stock loader, then re-wrap.
                dense = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                weights = dense.state_dict()
        else:
            dense = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            weights = dense.state_dict()

        config = kwargs.get("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            config.do_layer_distill = False

        model = cls(config)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        expected_missing = {"layer_transformation.weight", "layer_transformation.bias"}
        unexpected_missing = [m for m in missing if m not in expected_missing]
        if unexpected_missing:
            logger.warning(f"Unexpected missing keys when loading dense weights: {unexpected_missing}")
        if unexpected:
            logger.warning(f"Unexpected extra keys when loading dense weights: {unexpected}")
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        inference: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
            inference=inference,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)
        if hidden_z is not None:
            pooled_output = pooled_output.mul(hidden_z)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

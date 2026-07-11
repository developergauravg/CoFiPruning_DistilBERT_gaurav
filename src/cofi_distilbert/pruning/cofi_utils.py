"""
Port of ``utils/cofi_utils.py`` from the official CoFiPruning repository,
adapted to DistilBERT's module layout.

This is where "soft" masking (differentiable ``zs`` tensors multiplied into
activations during CoFi training, see ``modeling_distilbert.py``) turns into
"hard", physical pruning: the functions below permanently resize the
underlying ``nn.Linear`` / ``nn.LayerNorm`` / ``nn.Embedding`` weight
tensors, the same two-step design the official implementation uses for
BERT/RoBERTa.

Path mapping from the official ``bert = model.bert`` accessor:

    BERT path                                          DistilBERT path
    ----------------------------------------------      -----------------------------------------------
    bert.encoder.layer[i].attention.self.query/key       distilbert.transformer.layer[i].attention.q_lin/k_lin
    bert.encoder.layer[i].attention.self.value            distilbert.transformer.layer[i].attention.v_lin
    bert.encoder.layer[i].attention.output.dense           distilbert.transformer.layer[i].attention.out_lin
    bert.encoder.layer[i].attention.output.LayerNorm        distilbert.transformer.layer[i].sa_layer_norm
    bert.encoder.layer[i].intermediate.dense                distilbert.transformer.layer[i].ffn.lin1
    bert.encoder.layer[i].output.dense                       distilbert.transformer.layer[i].ffn.lin2
    bert.encoder.layer[i].output.LayerNorm                    distilbert.transformer.layer[i].output_layer_norm
    bert.embeddings.word_embeddings/position_embeddings/       distilbert.embeddings.word_embeddings/position_embeddings
        token_type_embeddings                                     (DistilBERT has no token-type embeddings)
    bert.pooler.dense                                            model.pre_classifier (closest DistilBERT analogue)
    model.qa_outputs / model.classifier                            model.classifier
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import torch
from transformers import AutoConfig
from transformers.modeling_utils import prune_linear_layer

# ``keys`` here mirror the official ``utils/utils.py::calculate_parameters``:
# these submodules are excluded from the "prunable" parameter count because
# CoFi's L0 module's ``prunable_model_size`` also excludes them (embeddings,
# the optional layer-distillation projection, and the classification head).
_EXCLUDED_PARAM_NAME_FRAGMENTS = ("embedding", "layer_transformation", "classifier", "pre_classifier")


def calculate_parameters(module: torch.nn.Module) -> int:
    """Direct port of ``utils/utils.py::calculate_parameters``."""
    return sum(
        p.numel() for n, p in module.named_parameters() if not any(key in n for key in _EXCLUDED_PARAM_NAME_FRAGMENTS)
    )


def initialize_layer_transformation(model) -> None:
    """Port of the official helper: sets the layer-distillation projection to identity + zero bias."""
    model.layer_transformation.weight.data.copy_(torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)


# --------------------------------------------------------------------------
# Soft -> physical weight pre-multiplication (must happen before physically
# resizing tensors, otherwise pruning would discard the *unmasked* weight
# values instead of the trained, gate-scaled ones).
# --------------------------------------------------------------------------
def update_params(model, zs: Optional[Dict[str, torch.Tensor]]) -> None:
    """
    Port of the official ``update_params``. Multiplies weight tensors by
    their corresponding ``z`` values in place, so that the physical pruning
    step in ``prune_model_with_z`` (which only *removes* rows/columns, it
    never rescales them) discards exactly the same information the soft
    mask would have zeroed out during training.
    """
    if zs is None:
        return

    distilbert = model.distilbert
    config = model.config
    hidden_dims = config.dim
    num_heads = config.n_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.n_layers

    if "intermediate_z" in zs:
        for layer in range(num_layers):
            intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
            ffn = distilbert.transformer.layer[layer].ffn
            ffn.lin2.weight.data = ffn.lin2.weight.data.mul(intermediate_z)
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"][layer].cpu()
                ffn.lin2.weight.data = ffn.lin2.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
                ffn.lin2.bias.data = ffn.lin2.bias.data.mul(mlp_z)

    if "head_z" in zs:
        for layer in range(num_layers):
            head_z = zs["head_z"][layer].cpu().squeeze().clone()
            head_z = torch.repeat_interleave(head_z, dims_per_head)
            attn = distilbert.transformer.layer[layer].attention
            attn.v_lin.weight.data = attn.v_lin.weight.data.transpose(0, 1).mul(head_z).transpose(0, 1)
            attn.v_lin.bias.data = attn.v_lin.bias.data.mul(head_z)
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"][layer].cpu()
                attn.out_lin.weight.data = attn.out_lin.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
                attn.out_lin.bias.data = attn.out_lin.bias.data.mul(head_layer_z)

    if "hidden_z" in zs:
        hidden_z = zs["hidden_z"].cpu().squeeze().clone()
        distilbert.embeddings.word_embeddings.weight.data = distilbert.embeddings.word_embeddings.weight.data.mul(hidden_z)
        distilbert.embeddings.position_embeddings.weight.data = distilbert.embeddings.position_embeddings.weight.data.mul(
            hidden_z
        )
        for layer in range(num_layers):
            block = distilbert.transformer.layer[layer]
            attn = block.attention
            attn.k_lin.weight.data = attn.k_lin.weight.data.mul(hidden_z)
            attn.q_lin.weight.data = attn.q_lin.weight.data.mul(hidden_z)
            attn.v_lin.weight.data = attn.v_lin.weight.data.mul(hidden_z)
            attn.out_lin.weight.data = attn.out_lin.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            attn.out_lin.bias.data = attn.out_lin.bias.data.mul(hidden_z)
            block.ffn.lin1.weight.data = block.ffn.lin1.weight.data.mul(hidden_z)
            block.ffn.lin2.weight.data = block.ffn.lin2.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
        if hasattr(model, "pre_classifier"):
            model.pre_classifier.weight.data = model.pre_classifier.weight.data.mul(hidden_z)
        if hasattr(model, "classifier"):
            model.classifier.weight.data = model.classifier.weight.data.mul(hidden_z)


# --------------------------------------------------------------------------
# Physical pruning
# --------------------------------------------------------------------------
def _prune_layer_norm(layernorm, index: torch.Tensor) -> None:
    layernorm.weight = torch.nn.parameter.Parameter(layernorm.weight.index_select(0, index))
    layernorm.bias = torch.nn.parameter.Parameter(layernorm.bias.index_select(0, index))
    layernorm.normalized_shape = (len(index),)


def prune_intermediate_layers(model, keep_dims: Dict[int, list]) -> None:
    """
    Physical FFN pruning: port of the official ``prune_intermediate_layers``.
    ``keep_dims[layer]`` is the list of intermediate (``lin1``/``lin2``)
    indices to keep for that layer; an empty list physically removes the
    entire FFN sublayer (``CoFiFFN.forward`` already knows how to skip a
    ``None`` ``lin1``).
    """
    distilbert = model.distilbert
    device = model.device
    for layer in keep_dims:
        ffn = distilbert.transformer.layer[layer].ffn
        if len(keep_dims[layer]) == 0:
            ffn.lin1 = None
            ffn.lin2 = None
        else:
            index = torch.LongTensor(keep_dims[layer]).to(device)
            ffn.lin1 = prune_linear_layer(ffn.lin1, index=index, dim=0)
            ffn.lin2 = prune_linear_layer(ffn.lin2, index=index, dim=1)


def prune_model_with_z(zs: Optional[Dict[str, torch.Tensor]], model) -> None:
    """
    Physical structural pruning: port of the official ``prune_model_with_z``.
    Performs, in order:
        1. Physical attention head pruning (``head_z`` / ``head_layer_z``)
        2. Physical FFN pruning (``intermediate_z`` / ``mlp_z``)
        3. Physical hidden-dimension pruning (``hidden_z``), which touches
           every remaining weight tensor in the model (embeddings, every
           q/k/v/out_lin, every ffn.lin1/lin2, layer norms, pre_classifier,
           classifier, and the optional layer_transformation projection).
    """
    if zs is None:
        return None

    distilbert = model.distilbert

    if "head_z" in zs:
        head_z = zs.get("head_z")
        head_layer_z = zs.get("head_layer_z")

        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer = head_z_layer * head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
            print(f"Layer {layer}, heads {' '.join(str(i) for i in index)} pruned.")
        distilbert._prune_heads(prune_heads)

    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z")
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze().cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer = intermediate_z_layer * mlp_z[layer]
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()

    if "hidden_z" in zs:
        hidden_zs = zs["hidden_z"]
        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze(-1).tolist())
        index = index.to(model.device)

        distilbert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
            distilbert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach()
        )
        distilbert.embeddings.word_embeddings.embedding_dim = index.shape[0]
        distilbert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
            distilbert.embeddings.position_embeddings.weight.index_select(1, index).clone().detach()
        )
        distilbert.embeddings.position_embeddings.embedding_dim = index.shape[0]
        _prune_layer_norm(distilbert.embeddings.LayerNorm, index)

        num_layers = model.config.n_layers
        for layer in range(num_layers):
            block = distilbert.transformer.layer[layer]
            attn = block.attention
            if attn.q_lin is not None:
                attn.q_lin = prune_linear_layer(attn.q_lin, index, dim=1)
                attn.k_lin = prune_linear_layer(attn.k_lin, index, dim=1)
            if attn.v_lin is not None:
                attn.v_lin = prune_linear_layer(attn.v_lin, index, dim=1)
                attn.out_lin = prune_linear_layer(attn.out_lin, index, dim=0)
                _prune_layer_norm(block.sa_layer_norm, index)
            if block.ffn.lin1 is not None:
                block.ffn.lin1 = prune_linear_layer(block.ffn.lin1, index, dim=1)
                block.ffn.lin2 = prune_linear_layer(block.ffn.lin2, index, dim=0)
                _prune_layer_norm(block.output_layer_norm, index)

        if hasattr(model, "pre_classifier"):
            # pre_classifier is Linear(dim, dim): only its *input* side consumes the
            # global hidden dimension that was just pruned. Its output side stays at
            # the original width -- mirroring the official code's handling of
            # bert.pooler.dense, which is likewise only pruned along dim=1. This is
            # why model.classifier below is intentionally left untouched: its input
            # (pre_classifier's output) never actually shrank.
            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(model.layer_transformation, index, dim=1)
            print("layer_transformation", model.layer_transformation.weight.shape)

    if kept_intermediate_dims is not None:
        prune_intermediate_layers(model, kept_intermediate_dims)

    num_layers = model.config.n_layers
    for layer in range(num_layers):
        block = distilbert.transformer.layer[layer]
        print("Layer:", layer)
        if block.attention.q_lin is not None:
            print("q_lin:", block.attention.q_lin.weight.shape)
            print("k_lin:", block.attention.k_lin.weight.shape)
        else:
            print("q_lin:", None)
            print("k_lin:", None)
        if block.attention.v_lin is not None:
            print("v_lin:", block.attention.v_lin.weight.shape)
            print("out_lin:", block.attention.out_lin.weight.shape)
        else:
            print("v_lin:", None)
            print("out_lin:", None)
        if block.ffn.lin1 is not None:
            print("lin1:", block.ffn.lin1.weight.shape)
            print("lin2:", block.ffn.lin2.weight.shape)
        else:
            print("lin1:", None)
            print("lin2:", None)


# --------------------------------------------------------------------------
# Compact model generation, export, loading
# --------------------------------------------------------------------------
def generate_compact_model(model, zs: Optional[Dict[str, torch.Tensor]]):
    """
    Compact model generation. Deep-copies ``model`` (never mutates the
    trained soft-masked model in place, so the caller keeps a full-size
    copy for further training/evaluation if needed), applies
    ``update_params`` to bake the gate values into the surviving weights,
    then physically prunes via ``prune_model_with_z``. Returns a smaller
    ``nn.Module`` whose forward pass no longer needs any ``zs`` argument.
    """
    compact_model = copy.deepcopy(model)
    print(f"Model size before pruning: {calculate_parameters(compact_model)}")
    update_params(compact_model, zs)
    prune_model_with_z(zs, compact_model)
    print(f"Model size after pruning: {calculate_parameters(compact_model)}")
    return compact_model


def export_compact_model(model, output_dir: str, zs: Optional[Dict[str, torch.Tensor]] = None) -> str:
    """
    Compact model export. Because a physically-pruned checkpoint has a
    different tensor shape per layer (variable head/FFN/hidden counts),
    it cannot be reloaded through ``config.json``'s single scalar
    ``n_heads``/``hidden_dim``/``dim`` fields the way a dense model can.
    Following the official implementation's approach, we therefore save:
        - ``pytorch_model.bin``: the pruned ``state_dict``
        - ``config.json``: the *original, dense* config (used only to
          recover static hyperparameters like ``vocab_size``,
          ``num_labels``, activation function, etc.)
        - ``zs.pt`` (optional): the hard zs used to prune, so
          ``load_compact_model`` can skip shape inference if desired
    ``load_compact_model`` below reconstructs a dense model from
    ``config.json``, then infers each layer's actual shape directly from
    the saved weight tensors -- exactly mirroring the official
    ``load_pruned_model``.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    if zs is not None:
        torch.save(zs, os.path.join(output_dir, "zs.pt"))
    print(f"Exported compact model to {output_dir}")
    return output_dir


def load_zs(model_path: str):
    """Port of the official ``load_zs``."""
    zs_path = model_path if model_path.endswith("zs.pt") else os.path.join(model_path, "zs.pt")
    if os.path.exists(zs_path):
        return torch.load(zs_path, map_location="cpu")
    return None


def load_pruned_model(model, weights: Dict[str, torch.Tensor]):
    """
    Port of the official ``load_pruned_model``: infers a ``zs`` dict purely
    from the *shapes* of a saved (already pruned) state_dict, physically
    prunes a freshly constructed dense model down to match those shapes,
    then loads the weights in. This is what lets a compact checkpoint be
    reloaded without needing per-layer shape metadata anywhere else.
    """
    config = model.config
    dim_per_head = config.dim // config.n_heads
    zs = {}

    hidden_z = torch.zeros(config.dim)
    hidden_z[: weights["distilbert.embeddings.word_embeddings.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.n_layers, config.n_heads)
    head_layer_z = torch.zeros(config.n_layers)
    for i in range(config.n_layers):
        key = f"distilbert.transformer.layer.{i}.attention.out_lin.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.n_layers, config.hidden_dim)
    mlp_z = torch.zeros(config.n_layers)
    for i in range(config.n_layers):
        key = f"distilbert.transformer.layer.{i}.ffn.lin2.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z

    prune_model_with_z(zs, model)
    model.load_state_dict(weights, strict=False)
    return model


def load_compact_model(model_path: str, model_class=None, num_labels: Optional[int] = None):
    """
    Compact model loading, entry point mirroring the official
    ``load_model`` / ``load_model_with_zs``. Reads ``config.json`` (the
    dense config saved by ``export_compact_model``), builds a fresh dense
    model from it, then calls ``load_pruned_model`` to shape-infer and
    physically prune it down to whatever shape the checkpoint actually is,
    before loading the pruned weights.
    """
    from cofi_distilbert.pruning.modeling_distilbert import CoFiDistilBertForSequenceClassification

    model_class = model_class or CoFiDistilBertForSequenceClassification

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_path}; was this exported with export_compact_model?")
    config = AutoConfig.from_pretrained(model_path)
    if num_labels is not None:
        config.num_labels = num_labels
    config.do_layer_distill = getattr(config, "do_layer_distill", False)

    model = model_class(config)

    weights_path = os.path.join(model_path, "pytorch_model.bin")
    loaded_weights = torch.load(weights_path, map_location="cpu")
    print(f"Loaded weights from {model_path}")

    print(f"Model size before pruning: {calculate_parameters(model)}")
    load_pruned_model(model, loaded_weights)
    print(f"Model size after pruning: {calculate_parameters(model)}")
    return model


def get_full_model_size(model_class, model_name: str) -> int:
    """Port of the official ``get_full_model_size``."""
    model = model_class.from_pretrained(model_name)
    return calculate_parameters(model)

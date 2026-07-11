"""
CoFi structural (physical) pruning for DistilBERT.

This subpackage is a port of the official Princeton-NLP CoFiPruning
implementation (originally written for BERT/RoBERTa) to DistilBERT. It is
additive to the rest of the ``cofi_distilbert`` package: nothing here is
imported by ``run_sweep.py``, ``cofi.py``, ``data.py`` or ``metrics.py``, so
the existing activation-masking training pipeline is unaffected unless a
caller explicitly opts in to this module.

Where the existing ``cofi_distilbert.cofi`` module implements *activation
masking* (forward hooks that zero out activations at inference time, leaving
every weight tensor at its original size), this subpackage implements *real*
structural pruning: attention heads, FFN intermediate neurons, and (for
symmetry with the full CoFi feature set) whole hidden-state columns are
physically removed from the weight tensors, so the resulting checkpoint is
smaller on disk and faster to run.

Public API:
    - ``config_shim.load_cofi_config``: build a DistilBertConfig with the
      extra attributes CoFi's model classes expect.
    - ``l0_module.DistilBertL0Module``: Hard-Concrete L0 gate module that
      produces the same ``zs`` dict keys as the official BERT L0Module
      (``hidden_z``, ``head_z``, ``head_layer_z``, ``intermediate_z``,
      ``mlp_z``).
    - ``modeling_distilbert.CoFiDistilBertForSequenceClassification``: a
      DistilBERT classifier whose forward pass accepts those ``zs`` for
      differentiable soft masking during CoFi training.
    - ``cofi_utils``: physical head/FFN/hidden pruning, compact model
      generation, export, and loading.
"""

from cofi_distilbert.pruning.config_shim import edit_config, load_cofi_config
from cofi_distilbert.pruning.l0_module import DistilBertL0Module
from cofi_distilbert.pruning.modeling_distilbert import (
    CoFiDistilBertForSequenceClassification,
    CoFiDistilBertModel,
)
from cofi_distilbert.pruning.cofi_utils import (
    calculate_parameters,
    generate_compact_model,
    export_compact_model,
    load_compact_model,
    load_zs,
    prune_model_with_z,
    update_params,
)

__all__ = [
    "edit_config",
    "load_cofi_config",
    "DistilBertL0Module",
    "CoFiDistilBertForSequenceClassification",
    "CoFiDistilBertModel",
    "calculate_parameters",
    "generate_compact_model",
    "export_compact_model",
    "load_compact_model",
    "load_zs",
    "prune_model_with_z",
    "update_params",
]

"""
Port of ``utils/cofi_utils.py::edit_config`` from the official CoFiPruning
repository, adapted for ``DistilBertConfig``.

The official function stamps a handful of extra attributes onto a
``BertConfig`` so the CoFi model classes (``CoFiBertForSequenceClassification``
etc.) know whether to build a layer-distillation projection head and whether
distillation is enabled at all:

    def edit_config(config, additional_args):
        config.transform_embedding = additional_args.transform_embedding
        config.do_distill = additional_args.do_distill
        config.do_layer_distill = additional_args.do_layer_distill

DistilBERT's config object uses different field names for the core
dimensions (``dim`` / ``hidden_dim`` / ``n_heads`` / ``n_layers`` instead of
BERT's ``hidden_size`` / ``intermediate_size`` / ``num_attention_heads`` /
``num_hidden_layers``), so in addition to the port of ``edit_config`` this
module provides a small set of read-only aliases (``required_l0_fields``)
used by ``l0_module.py`` to validate a config before building gates from it,
and a convenience constructor, ``load_cofi_config``, that produces a fully
prepared config in one call (mirroring the ``AutoConfig.from_pretrained`` +
``edit_config`` pattern used throughout the official ``run_glue_prune.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig, DistilBertConfig

# The four dimension fields the L0 module and the CoFi model classes need.
# DistilBERT names them differently than BERT; we keep both names visible
# here so anyone porting further CoFi code has a single place to look up
# the mapping.
REQUIRED_L0_FIELDS = ("dim", "hidden_dim", "n_heads", "n_layers")

BERT_TO_DISTILBERT_FIELD_MAP = {
    "hidden_size": "dim",
    "intermediate_size": "hidden_dim",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
}


@dataclass
class CoFiAdditionalArgs:
    """
    Stand-in for the ``additional_args`` Namespace the official
    ``edit_config`` expects (normally produced by ``args.py``'s
    ``AdditionalArguments`` dataclass). Only the fields CoFi's model
    classes actually branch on are kept.
    """

    do_distill: bool = True
    do_layer_distill: bool = False
    transform_embedding: bool = False


def edit_config(config: DistilBertConfig, additional_args: CoFiAdditionalArgs) -> DistilBertConfig:
    """
    Direct port of the official ``edit_config``. Stamps the CoFi-specific
    flags onto ``config`` in place and returns it for convenience.
    """
    config.transform_embedding = additional_args.transform_embedding
    config.do_distill = additional_args.do_distill
    config.do_layer_distill = additional_args.do_layer_distill
    return config


def validate_l0_fields(config: DistilBertConfig) -> None:
    """
    Raise a clear error early if ``config`` is missing a field the L0
    module or the CoFi model classes rely on, instead of failing deep
    inside a tensor-shape mismatch later.
    """
    missing = [f for f in REQUIRED_L0_FIELDS if not hasattr(config, f)]
    if missing:
        raise AttributeError(
            "DistilBertL0Module requires the DistilBERT-style config fields "
            f"{REQUIRED_L0_FIELDS}, but this config is missing {missing}. "
            "If you loaded a BERT config by mistake, note the field mapping: "
            f"{BERT_TO_DISTILBERT_FIELD_MAP}."
        )


def load_cofi_config(
    model_name_or_path: str,
    num_labels: int,
    do_layer_distill: bool = False,
    transform_embedding: bool = False,
    do_distill: bool = True,
    output_hidden_states: bool = True,
) -> DistilBertConfig:
    """
    Convenience constructor mirroring the
    ``AutoConfig.from_pretrained(...) + edit_config(...)`` pattern used in
    the official ``run_glue_prune.py``. Also forces
    ``output_hidden_states=True`` because the existing hidden-state
    distillation loss in ``cofi_distilbert.cofi.distillation_loss`` reads
    ``student_outputs.hidden_states`` / ``teacher_outputs.hidden_states``.
    """
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        output_hidden_states=output_hidden_states,
    )
    additional_args = CoFiAdditionalArgs(
        do_distill=do_distill,
        do_layer_distill=do_layer_distill,
        transform_embedding=transform_embedding,
    )
    edit_config(config, additional_args)
    validate_l0_fields(config)
    return config

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

LIMIT_A = -0.1
LIMIT_B = 1.1
EPS = 1e-6


class HardConcreteGate(nn.Module):
    def __init__(self, shape: tuple[int, ...], droprate_init: float = 0.5, temperature: float = 2.0 / 3.0):
        super().__init__()
        keep_prob = 1.0 - droprate_init
        init = math.log(keep_prob) - math.log(droprate_init)
        self.log_alpha = nn.Parameter(torch.empty(shape).normal_(mean=init, std=1e-2))
        self.temperature = temperature

    def cdf(self, x: float) -> torch.Tensor:
        xn = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)
        logits = math.log(xn) - math.log(1.0 - xn)
        return torch.sigmoid(logits * self.temperature - self.log_alpha).clamp(min=EPS, max=1.0 - EPS)

    def expected_l0(self) -> torch.Tensor:
        return 1.0 - self.cdf(0.0)

    def forward(self, training: bool = True) -> torch.Tensor:
        if training:
            u = torch.empty_like(self.log_alpha).uniform_(EPS, 1.0 - EPS)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.temperature)
            s = s * (LIMIT_B - LIMIT_A) + LIMIT_A
            return s.clamp(0.0, 1.0)
        return self.expected_l0().clamp(0.0, 1.0)

    def hard(self) -> torch.Tensor:
        return (self.expected_l0() >= 0.5).float()


@dataclass
class SparsityStats:
    head_sparsity: float
    ffn_sparsity: float
    total_sparsity: float
    effective_params: float
    full_prunable_params: float


class DistilBertL0Module(nn.Module):
    def __init__(
        self,
        config,
        droprate_init: float = 0.5,
        temperature: float = 2.0 / 3.0,
        target_sparsity: float = 0.0,
    ):
        super().__init__()
        self.num_layers = config.n_layers
        self.num_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dim_per_head = self.dim // self.num_heads
        self.target_sparsity = target_sparsity

        self.head_gate = HardConcreteGate((self.num_layers, self.num_heads), droprate_init, temperature)
        self.ffn_gate = HardConcreteGate((self.num_layers, self.hidden_dim), droprate_init, temperature)

        self.params_per_head = (4 * self.dim * self.dim + 4 * self.dim) / self.num_heads
        self.params_per_ffn_dim = (2 * self.dim) + 1
        self.full_prunable_params = (
            self.num_layers * self.num_heads * self.params_per_head
            + self.num_layers * self.hidden_dim * self.params_per_ffn_dim
        )

    def gates(self, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head_gate(training), self.ffn_gate(training)

    def expected_params(self) -> torch.Tensor:
        head_keep = self.head_gate.expected_l0()
        ffn_keep = self.ffn_gate.expected_l0()
        return head_keep.sum() * self.params_per_head + ffn_keep.sum() * self.params_per_ffn_dim

    def expected_sparsity(self) -> torch.Tensor:
        return 1.0 - self.expected_params() / self.full_prunable_params

    def hard_stats(self) -> SparsityStats:
        head_keep = self.head_gate.hard()
        ffn_keep = self.ffn_gate.hard()
        effective_params = (
            head_keep.sum().item() * self.params_per_head + ffn_keep.sum().item() * self.params_per_ffn_dim
        )
        return SparsityStats(
            head_sparsity=1.0 - head_keep.mean().item(),
            ffn_sparsity=1.0 - ffn_keep.mean().item(),
            total_sparsity=1.0 - effective_params / self.full_prunable_params,
            effective_params=effective_params,
            full_prunable_params=self.full_prunable_params,
        )

    def lagrangian_loss(self, step: int, warmup_steps: int, gamma: float) -> torch.Tensor:
        progress = min(1.0, step / max(1, warmup_steps))
        target = self.target_sparsity * progress
        diff = self.expected_sparsity() - target
        return gamma * diff.pow(2)

    def clamp_parameters(self) -> None:
        for gate in (self.head_gate, self.ffn_gate):
            gate.log_alpha.data.clamp_(min=math.log(1e-2), max=math.log(1e2))


class DistilBertGateHooks:
    def __init__(self, model: nn.Module, l0_module: DistilBertL0Module, hard: bool = False):
        self.model = model
        self.l0_module = l0_module
        self.hard = hard
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        layers = self.model.distilbert.transformer.layer
        for layer_idx, layer in enumerate(layers):
            self.handles.append(layer.attention.out_lin.register_forward_pre_hook(self._head_hook(layer_idx)))
            self.handles.append(layer.ffn.lin2.register_forward_pre_hook(self._ffn_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _head_hook(self, layer_idx: int):
        def hook(_module, inputs):
            (x,) = inputs
            if self.hard:
                gate = self.l0_module.head_gate.hard()[layer_idx].to(x.device)
            else:
                gate = self.l0_module.head_gate.expected_l0()[layer_idx].to(x.device)
            gate = gate.repeat_interleave(self.l0_module.dim_per_head).view(1, 1, -1)
            return (x * gate,)

        return hook

    def _ffn_hook(self, layer_idx: int):
        def hook(_module, inputs):
            (x,) = inputs
            if self.hard:
                gate = self.l0_module.ffn_gate.hard()[layer_idx].to(x.device)
            else:
                gate = self.l0_module.ffn_gate.expected_l0()[layer_idx].to(x.device)
            return (x * gate.view(1, 1, -1),)

        return hook


def distillation_loss(
    student_outputs,
    teacher_outputs,
    labels: torch.Tensor,
    alpha: float,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_outputs.logits, labels)
    kl = F.kl_div(
        F.log_softmax(student_outputs.logits / temperature, dim=-1),
        F.softmax(teacher_outputs.logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)

    hidden_loss = torch.tensor(0.0, device=labels.device)
    if student_outputs.hidden_states is not None and teacher_outputs.hidden_states is not None:
        pairs = zip(student_outputs.hidden_states, teacher_outputs.hidden_states)
        hidden_loss = torch.stack([F.mse_loss(s, t.detach()) for s, t in pairs]).mean()

    return ce + alpha * kl + beta * hidden_loss

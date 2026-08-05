from __future__ import annotations

import torch
from torch import nn

from Prior_Recon.Masked_Flow.trainer_masked_flow import (
    build_adamw_param_groups,
    migrate_single_group_optimizer_state,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(4, 12))
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, 4))
        self.look_invalid_emb = nn.Parameter(torch.zeros(1, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.ada_ln(x).chunk(3, dim=-1)[2]
        return self.norm(self.linear(x)) * gate + (self.pos_emb + self.look_invalid_emb).sum()


def _names(model: nn.Module, params: list[nn.Parameter]) -> set[str]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {by_id[id(p)] for p in params}


def test_gates_norms_biases_and_tables_are_excluded_from_weight_decay() -> None:
    model = _TinyModel()

    groups = build_adamw_param_groups(model, weight_decay=0.1)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert _names(model, groups[0]["params"]) == {"linear.weight", "ada_ln.1.weight"}
    assert _names(model, groups[1]["params"]) == {
        "linear.bias",
        "norm.weight",
        "norm.bias",
        # carries the residual gates -- decaying it shuts branches off
        "ada_ln.1.bias",
        "pos_emb",
        "look_invalid_emb",
    }
    # every parameter lands in exactly one group
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_single_group_optimizer_state_migrates_moments_exactly() -> None:
    torch.manual_seed(3)
    model = _TinyModel()
    legacy = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    model(torch.randn(2, 3, 4)).sum().backward()
    legacy.step()
    legacy_state = legacy.state_dict()
    moments_by_name = {
        name: legacy.state[param]["exp_avg"].clone()
        for name, param in model.named_parameters()
    }

    split = torch.optim.AdamW(
        build_adamw_param_groups(model, 0.1), lr=1e-3, weight_decay=0.1
    )
    migrated = migrate_single_group_optimizer_state(model, split, legacy_state)
    split.load_state_dict(migrated)

    assert [g["weight_decay"] for g in split.param_groups] == [0.1, 0.0]
    for name, param in model.named_parameters():
        assert torch.equal(split.state[param]["exp_avg"], moments_by_name[name])
        assert split.state[param]["step"] == legacy.state[param]["step"]


def test_matching_group_count_is_passed_through_unchanged() -> None:
    model = _TinyModel()
    split = torch.optim.AdamW(
        build_adamw_param_groups(model, 0.1), lr=1e-3, weight_decay=0.1
    )
    state = split.state_dict()

    assert migrate_single_group_optimizer_state(model, split, state) is state


def test_mismatched_parameter_set_reports_failure_instead_of_guessing() -> None:
    model = _TinyModel()
    legacy = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = legacy.state_dict()
    state["param_groups"][0]["params"] = state["param_groups"][0]["params"][:-1]
    split = torch.optim.AdamW(build_adamw_param_groups(model, 0.1), lr=1e-3)

    assert migrate_single_group_optimizer_state(model, split, state) is None

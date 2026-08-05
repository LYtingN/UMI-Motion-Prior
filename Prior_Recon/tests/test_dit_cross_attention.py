from __future__ import annotations

import pytest
import torch

from Prior_Recon.Masked_Flow.config import (
    EEMaskedFlowConfig,
    MotionRepConfig,
    SkeletonConfig,
)
from Prior_Recon.Masked_Flow.configs.load_config import config_from_yaml
from Prior_Recon.Masked_Flow.model.masked_flow_transformer import (
    EEMaskedFlowTransformer,
    load_masked_flow_state_dict,
)
from Prior_Recon.Masked_Flow.model.two_stage_cascade import (
    BodyStageTransformer,
)
from Prior_Recon.Masked_Flow.model.temporal_dit import (
    N_ADA_LN_CHUNKS,
    TemporalDiTConfigError,
    TemporalDiTSpec,
    TemporalDiTTransformer,
)
from Prior_Recon.Masked_Flow.model.temporal_dit_cross_attention import (
    NormalizedTemporalCrossAttention,
)


def _tiny_dit_config(**overrides) -> EEMaskedFlowConfig:
    kwargs = dict(
        skeleton=SkeletonConfig(
            name="tiny",
            n_total_joints=6,
            upper_body_global_indices=list(range(6)),
            wrist_local_indices=[],
        ),
        motion=MotionRepConfig(
            joint_repr="dof",
            seq_len=4,
            fps=30,
            window_stride=2,
            normalize=False,
        ),
        hidden_dim=16,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        time_emb_dim=8,
        transformer_ffn_mult=3,
        out_proj_hidden_mult=1,
        temporal_backbone="dit",
        use_ee_vel=True,
        lookahead_len=4,
        lookahead_stride=2,
        dit_residual_gate_init=0.1,
        dit_mask_invalid_lookahead=True,
    )
    kwargs.update(overrides)
    return EEMaskedFlowConfig(**kwargs)


def _tiny_spec(**overrides) -> TemporalDiTSpec:
    kwargs = dict(
        hidden_dim=16,
        n_heads=4,
        ffn_mult=3,
        dropout=0.0,
        n_layers=2,
    )
    kwargs.update(overrides)
    return TemporalDiTSpec(**kwargs)


def test_dit_config_has_no_legacy_attention_switch() -> None:
    cfg = config_from_yaml(
        "Prior_Recon/Masked_Flow/configs/delta73_dit_footfix.yaml"
    )

    assert not hasattr(cfg, "dit_use_cross_attention")
    assert not hasattr(cfg, "dit_cross_attention_gate_init")
    assert cfg.dit_residual_gate_init == 0.1
    assert cfg.dit_mask_invalid_lookahead is True


def test_dit_cross_attention_receives_ee_gradient_when_head_is_open() -> None:
    torch.manual_seed(17)
    cfg = _tiny_dit_config()
    model = EEMaskedFlowTransformer(cfg)
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    known = torch.zeros_like(x_t)
    mask = torch.zeros_like(x_t)
    mask[:, :1] = 1.0
    s_ee = torch.randn(2, 4, cfg.ee_feat_dim)
    target = torch.randn_like(x_t)

    output = model(x_t, known, mask, s_ee, torch.rand(2))
    loss = (output.pred_v - target).square().mean()
    loss.backward()

    ee_gradient = model.ee_proj[0].weight.grad
    assert ee_gradient is not None
    assert torch.linalg.vector_norm(ee_gradient) > 0.0


def test_invalid_dit_lookahead_payload_does_not_change_prediction() -> None:
    torch.manual_seed(29)
    cfg = _tiny_dit_config()
    model = EEMaskedFlowTransformer(cfg).eval()
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    known = torch.zeros_like(x_t)
    mask = torch.zeros_like(x_t)
    s_ee = torch.randn(2, 4, cfg.ee_feat_dim)
    t = torch.full((2,), 0.5)
    reference_lookahead = torch.randn(2, 4, cfg.ee_feat_dim)
    look_valid = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
    )
    adversarial_lookahead = reference_lookahead.clone()
    adversarial_lookahead[~look_valid.bool()] = torch.nan

    reference = model(
        x_t, known, mask, s_ee, t,
        s_ee_look=reference_lookahead, look_valid=look_valid,
    ).pred_v
    adversarial = model(
        x_t, known, mask, s_ee, t,
        s_ee_look=adversarial_lookahead, look_valid=look_valid,
    ).pred_v

    assert torch.isfinite(adversarial).all()
    assert torch.equal(reference, adversarial)


def test_dit_ignores_invalid_lookahead_tokens_when_preview_is_truncated() -> None:
    # Given: a DiT with masking enabled receives a completely truncated preview.
    torch.manual_seed(31)
    cfg = _tiny_dit_config()
    model = EEMaskedFlowTransformer(cfg).eval()
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    known = torch.zeros_like(x_t)
    mask = torch.zeros_like(x_t)
    s_ee = torch.randn(2, 4, cfg.ee_feat_dim)
    t = torch.full((2,), 0.5)
    lookahead = torch.randn(2, 4, cfg.ee_feat_dim)
    look_valid = torch.zeros(2, 4)

    reference = model(
        x_t, known, mask, s_ee, t,
        s_ee_look=lookahead, look_valid=look_valid,
    ).pred_v

    # When: the preview payload of those invalid tokens changes arbitrarily.
    changed = model(
        x_t, known, mask, s_ee, t,
        s_ee_look=lookahead + 5.0, look_valid=look_valid,
    ).pred_v

    # Then: masked-out tokens cannot participate in attention.
    assert torch.isfinite(changed).all()
    assert torch.equal(reference, changed)
    # And: the "invalid" marker embedding is not built at all in this mode.
    assert not hasattr(model, "look_invalid_emb")


def test_marker_mode_keeps_the_learned_invalid_embedding_live() -> None:
    # The legacy convention (dit_mask_invalid_lookahead=False) tags invalid
    # preview tokens instead of dropping them, so the marker MUST matter --
    # otherwise old checkpoints would be silently reinterpreted.
    torch.manual_seed(37)
    cfg = _tiny_dit_config(dit_mask_invalid_lookahead=False)
    model = EEMaskedFlowTransformer(cfg).eval()
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    known = torch.zeros_like(x_t)
    mask = torch.zeros_like(x_t)
    s_ee = torch.randn(2, 4, cfg.ee_feat_dim)
    t = torch.full((2,), 0.5)
    lookahead = torch.randn(2, 4, cfg.ee_feat_dim)
    look_valid = torch.zeros(2, 4)
    kwargs = dict(s_ee_look=lookahead, look_valid=look_valid)

    reference = model(x_t, known, mask, s_ee, t, **kwargs).pred_v
    with torch.no_grad():
        model.look_invalid_emb.fill_(100.0)
    changed = model(x_t, known, mask, s_ee, t, **kwargs).pred_v

    assert not torch.equal(reference, changed)


def test_lookahead_has_its_own_projection() -> None:
    cfg = _tiny_dit_config()
    model = EEMaskedFlowTransformer(cfg)

    assert model.look_proj is not model.ee_proj
    assert all(
        look is not ee
        for look, ee in zip(model.look_proj.parameters(), model.ee_proj.parameters())
    )


def test_dit_cross_attention_normalizes_context_scale() -> None:
    torch.manual_seed(23)
    model = TemporalDiTTransformer(_tiny_spec(residual_gate_init=0.1))
    body = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)

    reference = model(body, context, condition)
    scaled = model(body, context * 10.0, condition)

    assert torch.allclose(reference, scaled, atol=2e-5, rtol=2e-5)


def test_every_dit_block_uses_dedicated_cross_attention() -> None:
    torch.manual_seed(7)
    model = TemporalDiTTransformer(_tiny_spec())
    body = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)

    output = model(body, context, condition)

    assert output.shape == body.shape
    for block in model.blocks:
        assert isinstance(
            block.context_attention,
            NormalizedTemporalCrossAttention,
        )
        # The cross-attention no longer owns a standalone scalar gate; its gate
        # is chunk 5 of the block's adaLN head, so it can vary with t.
        assert not hasattr(block.context_attention, "gate")
        assert block.ada_ln[-1].out_features == N_ADA_LN_CHUNKS * 16


def test_dit_cross_attention_modulation_is_independent_of_self_attention() -> None:
    torch.manual_seed(11)
    model = TemporalDiTTransformer(_tiny_spec())
    block = model.blocks[0]
    with torch.no_grad():
        torch.nn.init.normal_(block.ada_ln[-1].weight, std=0.05)
    condition = torch.randn(2, 16)

    chunks = block.ada_ln(condition).chunk(N_ADA_LN_CHUNKS, dim=-1)

    assert not torch.allclose(chunks[3], chunks[0])
    assert not torch.allclose(chunks[4], chunks[1])


def test_dit_gates_start_warm_but_modulation_stays_data_independent() -> None:
    model = TemporalDiTTransformer(_tiny_spec(residual_gate_init=0.1))

    for block in model.blocks:
        head = block.ada_ln[-1]
        # Zero weight => shift/scale/gate do not depend on the condition yet.
        assert torch.equal(head.weight, torch.zeros_like(head.weight))
        chunks = head.bias.chunk(N_ADA_LN_CHUNKS)
        for gate_chunk in (2, 5, 8):
            assert torch.equal(
                chunks[gate_chunk], torch.full_like(chunks[gate_chunk], 0.1)
            )
        for passthrough_chunk in (0, 1, 3, 4, 6, 7):
            assert torch.equal(
                chunks[passthrough_chunk],
                torch.zeros_like(chunks[passthrough_chunk]),
            )


def test_canonical_zero_gate_is_reachable_and_is_the_identity() -> None:
    torch.manual_seed(13)
    model = TemporalDiTTransformer(_tiny_spec(residual_gate_init=0.0))
    body = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)

    output = model(body, context, condition)

    # Every residual branch is gated off, so the stack reduces to final_norm.
    assert torch.allclose(output, model.final_norm(body), atol=1e-6)


def test_out_of_range_gate_init_is_rejected() -> None:
    with pytest.raises(TemporalDiTConfigError):
        TemporalDiTTransformer(_tiny_spec(residual_gate_init=-0.1))
    with pytest.raises(TemporalDiTConfigError):
        TemporalDiTTransformer(_tiny_spec(residual_gate_init=1.5))


def test_all_masked_context_row_raises_instead_of_emitting_nan() -> None:
    torch.manual_seed(19)
    model = TemporalDiTTransformer(_tiny_spec())
    body = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)
    key_padding_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(TemporalDiTConfigError):
        model(body, context, condition, key_padding_mask)


def _legacy_dit_state_dict(
    model: EEMaskedFlowTransformer, static_gate: float
) -> dict[str, torch.Tensor]:
    """Fold the current state dict back into the pre-fix on-disk layout."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    hidden = model.temporal_dit.hidden_dim
    for index, _ in enumerate(model.temporal_dit.blocks):
        prefix = f"temporal_dit.blocks.{index}"
        for suffix in ("weight", "bias"):
            key = f"{prefix}.ada_ln.1.{suffix}"
            chunks = list(state[key].chunk(N_ADA_LN_CHUNKS))
            # legacy order: attn shift/scale/gate then mlp shift/scale/gate.
            state[key] = torch.cat(
                [chunks[0], chunks[1], chunks[2], chunks[6], chunks[7], chunks[8]],
                dim=0,
            )
        state[f"{prefix}.context_attention.gate"] = torch.full(
            (hidden,), static_gate
        )
        state.pop(f"{prefix}.context_attention.norm_context.weight")
        state.pop(f"{prefix}.context_attention.norm_context.bias")
    for name in list(state):
        if name.startswith("look_proj."):
            state.pop(name)
    return state


def test_legacy_dit_checkpoint_migrates_to_legacy_semantics_exactly() -> None:
    torch.manual_seed(41)
    cfg = _tiny_dit_config(dit_mask_invalid_lookahead=False)
    source = EEMaskedFlowTransformer(cfg)
    with torch.no_grad():
        for block in source.temporal_dit.blocks:
            torch.nn.init.normal_(block.ada_ln[-1].weight, std=0.05)
            torch.nn.init.normal_(block.ada_ln[-1].bias, std=0.05)
    legacy = _legacy_dit_state_dict(source, static_gate=0.1)
    legacy_ada = {
        key: value.clone()
        for key, value in legacy.items()
        if key.endswith("ada_ln.1.weight") or key.endswith("ada_ln.1.bias")
    }

    target = EEMaskedFlowTransformer(cfg)
    load_masked_flow_state_dict(target, legacy)

    for index, block in enumerate(target.temporal_dit.blocks):
        prefix = f"temporal_dit.blocks.{index}"
        hidden = block.hidden_dim
        for suffix, chunks_expected in (
            ("weight", legacy_ada[f"{prefix}.ada_ln.1.weight"].chunk(6)),
            ("bias", legacy_ada[f"{prefix}.ada_ln.1.bias"].chunk(6)),
        ):
            new = getattr(block.ada_ln[-1], suffix).detach().chunk(N_ADA_LN_CHUNKS)
            # self-attention and mlp triples carry over untouched
            for new_index, old_index in ((0, 0), (1, 1), (2, 2), (6, 3), (7, 4), (8, 5)):
                assert torch.equal(new[new_index], chunks_expected[old_index])
            # cross-attention reused the SELF-attention shift/scale ...
            assert torch.equal(new[3], chunks_expected[0])
            assert torch.equal(new[4], chunks_expected[1])
        # ... and its gate was a condition-independent constant.
        gate_weight = block.ada_ln[-1].weight.detach().chunk(N_ADA_LN_CHUNKS)[5]
        gate_bias = block.ada_ln[-1].bias.detach().chunk(N_ADA_LN_CHUNKS)[5]
        assert torch.equal(gate_weight, torch.zeros_like(gate_weight))
        assert torch.equal(gate_bias, torch.full((hidden,), 0.1))
        # non-affine norm_context becomes affine with identity gains.
        norm = block.context_attention.norm_context
        assert torch.equal(norm.weight.detach(), torch.ones(hidden))
        assert torch.equal(norm.bias.detach(), torch.zeros(hidden))
    # the preview projection is seeded from the shared ee_proj it used to be
    for look, ee in zip(target.look_proj.parameters(), target.ee_proj.parameters()):
        assert torch.equal(look.detach(), ee.detach())


def test_legacy_dit_checkpoint_prediction_is_unchanged_after_migration() -> None:
    torch.manual_seed(43)
    cfg = _tiny_dit_config(dit_mask_invalid_lookahead=False)
    source = EEMaskedFlowTransformer(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(source.out_proj.weight, std=0.02)
        # A legacy checkpoint has no look_proj: the preview went through the
        # shared ee_proj, which is exactly what the migration seeds it with.
        for look, ee in zip(source.look_proj.parameters(), source.ee_proj.parameters()):
            look.copy_(ee)
        for block in source.temporal_dit.blocks:
            torch.nn.init.normal_(block.ada_ln[-1].weight, std=0.05)
            torch.nn.init.normal_(block.ada_ln[-1].bias, std=0.05)
            # legacy semantics: cross shift/scale ARE the self-attention ones
            head = block.ada_ln[-1]
            for tensor in (head.weight, head.bias):
                chunks = list(tensor.chunk(N_ADA_LN_CHUNKS))
                chunks[3].copy_(chunks[0])
                chunks[4].copy_(chunks[1])
                chunks[5].fill_(0.1)
            torch.nn.init.zeros_(
                head.weight.chunk(N_ADA_LN_CHUNKS)[5]
            )
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    known = torch.zeros_like(x_t)
    mask = torch.zeros_like(x_t)
    s_ee = torch.randn(2, 4, cfg.ee_feat_dim)
    t = torch.full((2,), 0.4)
    lookahead = torch.randn(2, 4, cfg.ee_feat_dim)
    look_valid = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    kwargs = dict(s_ee_look=lookahead, look_valid=look_valid)
    expected = source(x_t, known, mask, s_ee, t, **kwargs).pred_v

    target = EEMaskedFlowTransformer(cfg).eval()
    load_masked_flow_state_dict(
        target, _legacy_dit_state_dict(source, static_gate=0.1)
    )
    actual = target(x_t, known, mask, s_ee, t, **kwargs).pred_v

    assert torch.equal(expected, actual)


def test_pre_cross_attention_dit_checkpoint_loads_with_the_branch_off() -> None:
    # The oldest dit checkpoints have a 6-chunk adaLN and NO context branch at
    # all. Gating the new branch off reproduces them exactly.
    torch.manual_seed(47)
    cfg = _tiny_dit_config(dit_mask_invalid_lookahead=False)
    source = EEMaskedFlowTransformer(cfg).eval()
    legacy = _legacy_dit_state_dict(source, static_gate=0.0)
    for name in list(legacy):
        if ".context_attention." in name:
            legacy.pop(name)

    target = EEMaskedFlowTransformer(cfg).eval()
    load_masked_flow_state_dict(target, legacy)

    for block in target.temporal_dit.blocks:
        gate_weight = block.ada_ln[-1].weight.detach().chunk(N_ADA_LN_CHUNKS)[5]
        gate_bias = block.ada_ln[-1].bias.detach().chunk(N_ADA_LN_CHUNKS)[5]
        assert torch.equal(gate_weight, torch.zeros_like(gate_weight))
        assert torch.equal(gate_bias, torch.zeros_like(gate_bias))


def test_dit_backbone_drops_the_dead_pre_backbone_norm() -> None:
    dit_model = EEMaskedFlowTransformer(_tiny_dit_config())
    flat_model = EEMaskedFlowTransformer(
        _tiny_dit_config(temporal_backbone="flat")
    )

    assert dit_model.norm is None
    assert flat_model.norm is not None
    assert not any(key.startswith("norm.") for key in dit_model.state_dict())


def _cascade_config(**overrides) -> EEMaskedFlowConfig:
    cfg = _tiny_dit_config(**overrides)
    cfg.root_look_len = 4
    cfg.root_look_stride = 2
    cfg.n_root_look_tokens = 2
    return cfg


@pytest.mark.parametrize("backbone", ["dit", "flat"])
def test_body_stage_cascade_forward_survives_the_preview_plumbing(backbone: str) -> None:
    # BodyStageTransformer overrides forward() and re-implements the token
    # assembly, so every change to _lookahead_tokens / _encode_body_tokens has
    # to be mirrored there or the cascade breaks with no other test noticing.
    torch.manual_seed(53)
    cfg = _cascade_config(temporal_backbone=backbone)
    model = BodyStageTransformer(cfg, root_dim=6).eval()
    x_t = torch.randn(2, 4, cfg.n_total_dof)
    look_valid = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    output = model(
        x_t,
        torch.zeros_like(x_t),
        torch.zeros_like(x_t),
        torch.randn(2, 4, cfg.ee_feat_dim),
        torch.full((2,), 0.5),
        s_ee_look=torch.randn(2, cfg.lookahead_len, cfg.ee_feat_dim),
        look_valid=look_valid,
        root_look=torch.randn(2, cfg.root_look_len, 6),
        root_look_valid=torch.ones(2, cfg.root_look_len),
    )

    assert output.pred_v.shape == x_t.shape
    assert torch.isfinite(output.pred_v).all()


def test_cascade_root_preview_tokens_are_never_masked_out() -> None:
    # The root preview is appended AFTER the EE preview, so the EE-sized mask
    # must be extended -- otherwise it silently masks root tokens (or the
    # attention rejects the width).
    torch.manual_seed(59)
    cfg = _cascade_config()
    model = BodyStageTransformer(cfg, root_dim=6).eval()
    # The dit readout is zero-init, so open it or every prediction is 0.
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    root_look = torch.randn(2, cfg.root_look_len, 6)
    kwargs = dict(
        s_ee_look=torch.randn(2, cfg.lookahead_len, cfg.ee_feat_dim),
        look_valid=torch.zeros(2, cfg.lookahead_len),
        root_look_valid=torch.ones(2, cfg.root_look_len),
    )
    args = (
        torch.randn(2, 4, cfg.n_total_dof),
        torch.zeros(2, 4, cfg.n_total_dof),
        torch.zeros(2, 4, cfg.n_total_dof),
        torch.randn(2, 4, cfg.ee_feat_dim),
        torch.full((2,), 0.5),
    )

    reference = model(*args, root_look=root_look, **kwargs).pred_v
    changed = model(*args, root_look=root_look + 3.0, **kwargs).pred_v

    assert not torch.equal(reference, changed)

from __future__ import annotations

import torch

from Prior_Recon.Masked_Flow.config import (
    EEMaskedFlowConfig,
    MotionRepConfig,
    SkeletonConfig,
)
from Prior_Recon.Masked_Flow.configs.load_config import config_from_yaml
from Prior_Recon.Masked_Flow.model.masked_flow_transformer import (
    EEMaskedFlowTransformer,
)
from Prior_Recon.Masked_Flow.model.temporal_dit import (
    TemporalDiTSpec,
    TemporalDiTTransformer,
)
from Prior_Recon.Masked_Flow.model.temporal_dit_cross_attention import (
    NormalizedTemporalCrossAttention,
)


def _tiny_dit_config() -> EEMaskedFlowConfig:
    return EEMaskedFlowConfig(
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
        use_ee_vel=False,
        dit_cross_attention_gate_init=0.1,
    )


def test_dit_config_has_no_legacy_attention_switch() -> None:
    cfg = config_from_yaml(
        "Prior_Recon/Masked_Flow/configs/delta73_dit_footfix.yaml"
    )

    assert not hasattr(cfg, "dit_use_cross_attention")
    assert cfg.dit_cross_attention_gate_init == 0.1


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


def test_dit_cross_attention_normalizes_context_scale() -> None:
    torch.manual_seed(23)
    model = TemporalDiTTransformer(
        TemporalDiTSpec(
            hidden_dim=16,
            n_heads=4,
            ffn_mult=3,
            dropout=0.0,
            n_layers=2,
        )
    )
    body = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)

    reference = model(body, context, condition)
    scaled = model(body, context * 10.0, condition)

    assert torch.allclose(reference, scaled, atol=2e-5, rtol=2e-5)


def test_every_dit_block_uses_dedicated_cross_attention() -> None:
    torch.manual_seed(7)
    model = TemporalDiTTransformer(
        TemporalDiTSpec(
            hidden_dim=16,
            n_heads=4,
            ffn_mult=3,
            dropout=0.0,
            n_layers=2,
        )
    )
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
        assert torch.equal(
            block.context_attention.gate,
            torch.full_like(block.context_attention.gate, 0.1),
        )

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from Prior_Recon.Masked_Flow.config import PrimitiveConfig
from Prior_Recon.Masked_Flow.loss.foot_skate import (
    project_contact_probability,
)
from Prior_Recon.Masked_Flow.model.temporal_dit import (
    TemporalDiTSpec,
    TemporalDiTTransformer,
)
from Prior_Recon.Masked_Flow.model.temporal_pyramid import (
    TemporalPyramidConfigError,
    TemporalPyramidSpec,
    TemporalPyramidTransformer,
)


@dataclass
class MaskedFlowTransformerOutput:
    pred_v: torch.Tensor
    pred_x1: torch.Tensor
    x_in: torch.Tensor
    t: torch.Tensor


def build_prefix_condition_tensors(
    history_motion: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pin a history prefix and leave the remaining frames unknown."""
    if history_motion.ndim != 3:
        raise ValueError(
            f"history_motion must be rank-3 (B, H, D), got {tuple(history_motion.shape)}."
        )
    batch, history_len, feat_dim = history_motion.shape
    if history_len > seq_len:
        raise ValueError(
            f"history_len={history_len} exceeds seq_len={seq_len} for prefix conditioning."
        )

    known_full = torch.zeros(
        batch,
        seq_len,
        feat_dim,
        device=history_motion.device,
        dtype=history_motion.dtype,
    )
    obs_mask = torch.zeros_like(known_full)
    known_full[:, :history_len] = history_motion
    obs_mask[:, :history_len] = 1.0
    return known_full, obs_mask


def apply_ee_state_condition(
    known_full: torch.Tensor,
    obs_mask: torch.Tensor,
    ee_state: torch.Tensor,
) -> None:
    """Pin the trailing EE-state columns for every frame (in place).

    ``ee_state`` is (B, T, E); the last E feature dims of ``known_full`` become
    observed ground truth, so `_assemble_observed_state` re-imposes them at
    every denoising step — a hard constraint, same mechanism as the history
    prefix but along the feature axis.
    """
    if ee_state.ndim != 3:
        raise ValueError(f"ee_state must be rank-3 (B, T, E), got {tuple(ee_state.shape)}.")
    if ee_state.shape[:2] != known_full.shape[:2]:
        raise ValueError(
            f"ee_state shape {tuple(ee_state.shape)} must match known_full "
            f"{tuple(known_full.shape)} on (B, T)."
        )
    ee_dim = ee_state.shape[-1]
    if ee_dim <= 0 or ee_dim > known_full.shape[-1]:
        raise ValueError(f"Invalid ee_state dim {ee_dim} for feat dim {known_full.shape[-1]}.")
    known_full[..., -ee_dim:] = ee_state
    obs_mask[..., -ee_dim:] = 1.0


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Accepts t of shape (B,) -> (B, out_dim) or (B, T) -> (B, T, out_dim).
        # The frequency bank broadcasts on a trailing dim, so any leading shape
        # is preserved (needed for per-frame diffusion-forcing noise levels).
        half = self.emb_dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        angles = t.unsqueeze(-1) * freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return self.proj(emb)


class EEMaskedFlowTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        n_dof = cfg.n_total_dof
        ee_dim = cfg.ee_feat_dim
        hidden_dim = cfg.hidden_dim
        seq_len = cfg.motion.seq_len
        ffn_mult = int(getattr(cfg, "transformer_ffn_mult", 8))
        out_proj_hidden_mult = int(getattr(cfg, "out_proj_hidden_mult", 2))
        out_proj_hidden_dim = hidden_dim * max(out_proj_hidden_mult, 1)

        self.feat_proj = nn.Sequential(
            nn.Linear(n_dof, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mask_proj = nn.Linear(n_dof, hidden_dim, bias=False)
        nn.init.zeros_(self.mask_proj.weight)
        self.ee_pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.ee_pos_emb, std=0.02)

        self.ee_proj = nn.Sequential(
            nn.Linear(ee_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Frame-aligned EE injection: a SECOND, independent projection of the
        # per-frame condition that is added directly onto the body token at the
        # SAME time index (see forward). This turns the "which EE goes with which
        # body frame" alignment from a soft attention-routing problem (the
        # parallel ee_tokens, resolved only via query/key matching against
        # ee_pos_emb) into a hard tensor-index identity (body_tokens[:, t] gets
        # ee_frame_proj(s_ee)[:, t]). Attention is then freed to model temporal
        # structure instead of also having to retrieve the time-matched hand.
        self.ee_frame_proj = nn.Sequential(
            nn.Linear(ee_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # EE lookahead preview tokens: same feature format and projection as
        # the window condition, distinguished by their own positional slots.
        self.lookahead_len = int(getattr(cfg, "lookahead_len", 0))
        self.lookahead_stride = max(int(getattr(cfg, "lookahead_stride", 1)), 1)
        n_look = getattr(cfg, "n_lookahead_tokens", 0)
        if self.lookahead_len > 0 and not n_look:
            n_look = (self.lookahead_len + self.lookahead_stride - 1) // self.lookahead_stride
        self.n_lookahead_tokens = int(n_look) if self.lookahead_len > 0 else 0
        if self.n_lookahead_tokens > 0:
            self.look_pos_emb = nn.Parameter(
                torch.zeros(1, self.n_lookahead_tokens, hidden_dim)
            )
            nn.init.trunc_normal_(self.look_pos_emb, std=0.02)
            # Added to tokens whose validity is 0 (padding / truncated preview).
            # Zero-init: an all-invalid preview starts out indistinguishable
            # from plain positional tokens and the model learns the distinction.
            self.look_invalid_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.time_emb = SinusoidalTimeEmbedding(cfg.time_emb_dim, hidden_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.temporal_backbone = str(getattr(cfg, "temporal_backbone", "flat"))
        self.transformer: nn.TransformerEncoder | None = None
        self.temporal_pyramid: TemporalPyramidTransformer | None = None
        self.temporal_dit: TemporalDiTTransformer | None = None
        if self.temporal_backbone == "flat":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.n_heads,
                dim_feedforward=hidden_dim * ffn_mult,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer,
                num_layers=cfg.n_layers,
            )
        elif self.temporal_backbone == "hierarchical":
            spec = TemporalPyramidSpec(
                hidden_dim=hidden_dim,
                n_heads=cfg.n_heads,
                ffn_mult=ffn_mult,
                dropout=cfg.dropout,
                fine_layers=int(getattr(cfg, "hierarchy_fine_layers", 4)),
                coarse_layers=int(getattr(cfg, "hierarchy_coarse_layers", 4)),
                refine_layers=int(getattr(cfg, "hierarchy_refine_layers", 4)),
                downsample_factor=int(
                    getattr(cfg, "hierarchy_downsample_factor", 2)
                ),
            )
            self.temporal_pyramid = TemporalPyramidTransformer(spec)
        elif self.temporal_backbone == "dit":
            self.temporal_dit = TemporalDiTTransformer(
                TemporalDiTSpec(
                    hidden_dim=hidden_dim,
                    n_heads=cfg.n_heads,
                    ffn_mult=ffn_mult,
                    dropout=cfg.dropout,
                    n_layers=cfg.n_layers,
                )
            )
        else:
            raise TemporalPyramidConfigError(
                "temporal_backbone",
                self.temporal_backbone,
            )
        self.norm = nn.LayerNorm(hidden_dim)
        if out_proj_hidden_mult <= 1:
            self.out_proj = nn.Linear(hidden_dim, n_dof)
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, out_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(out_proj_hidden_dim, n_dof),
            )
        if self.temporal_backbone == "dit":
            output_linear = (
                self.out_proj
                if out_proj_hidden_mult <= 1
                else self.out_proj[2]
            )
            nn.init.zeros_(output_linear.weight)
            nn.init.zeros_(output_linear.bias)

    def _validate_condition(self, s_ee: torch.Tensor) -> None:
        expected = self.cfg.ee_feat_dim
        actual = s_ee.shape[-1]
        if actual != expected:
            raise ValueError(
                f"Expected s_ee last dim={expected}, got {actual}. "
                "Check cfg.use_ee_pos and dataset keypoint loading."
            )

    def _lookahead_tokens(
        self,
        s_ee: torch.Tensor,
        s_ee_look: torch.Tensor | None,
        look_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Preview tokens (B, n_look, hidden) from K raw lookahead frames.

        A missing preview degrades gracefully to the all-invalid case (window's
        last frame held still, validity 0) — the truncation-to-zero states seen
        in training, so a lookahead checkpoint can still run without a preview.
        """
        batch = s_ee.shape[0]
        if s_ee_look is None:
            hold = s_ee[:, -1:].clone()
            if getattr(self.cfg, "use_ee_vel", False):
                # Hold-still fallback: held reference => zero velocity, matching
                # the dataset's invalid-preview padding convention.
                hold[..., -18:] = 0.0
            s_ee_look = hold.expand(batch, self.lookahead_len, s_ee.shape[-1])
            look_valid = None
        if s_ee_look.shape[1] != self.lookahead_len or s_ee_look.shape[-1] != s_ee.shape[-1]:
            raise ValueError(
                f"s_ee_look must be (B, {self.lookahead_len}, {s_ee.shape[-1]}), "
                f"got {tuple(s_ee_look.shape)}."
            )
        if look_valid is None:
            look_valid = torch.zeros(
                batch, self.lookahead_len, device=s_ee.device, dtype=s_ee.dtype
            )
        if look_valid.shape[:2] != s_ee_look.shape[:2]:
            raise ValueError(
                f"look_valid shape {tuple(look_valid.shape)} must match "
                f"s_ee_look {tuple(s_ee_look.shape)} on (B, K)."
            )

        look_ds = s_ee_look[:, :: self.lookahead_stride][:, : self.n_lookahead_tokens]
        valid_ds = look_valid[:, :: self.lookahead_stride][:, : self.n_lookahead_tokens]
        tokens = self.ee_proj(look_ds) + self.look_pos_emb
        tokens = tokens + (1.0 - valid_ds.unsqueeze(-1)) * self.look_invalid_emb
        return tokens

    def _validate_observation_tensors(
        self,
        x_t: torch.Tensor,
        known_full: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> None:
        if known_full.shape != x_t.shape:
            raise ValueError(
                f"known_full shape {tuple(known_full.shape)} must match x_t shape {tuple(x_t.shape)}."
            )
        if obs_mask.shape != x_t.shape:
            raise ValueError(
                f"obs_mask shape {tuple(obs_mask.shape)} must match x_t shape {tuple(x_t.shape)}."
            )

    def _assemble_observed_state(
        self,
        x_t: torch.Tensor,
        known_full: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_observation_tensors(x_t, known_full, obs_mask)
        return obs_mask * known_full + (1.0 - obs_mask) * x_t

    def _encode_body_tokens(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.temporal_backbone == "flat":
            if self.transformer is None:
                raise TemporalPyramidConfigError("temporal_backbone", "flat")
            tokens = torch.cat((context_tokens, body_tokens), dim=1)
            return self.transformer(tokens)[:, -body_tokens.shape[1]:]
        if self.temporal_backbone == "hierarchical":
            if self.temporal_pyramid is None:
                raise TemporalPyramidConfigError(
                    "temporal_backbone",
                    "hierarchical",
                )
            return self.temporal_pyramid(body_tokens, context_tokens)
        if self.temporal_dit is None:
            raise TemporalPyramidConfigError(
                "temporal_backbone",
                "dit",
            )
        return self.temporal_dit(body_tokens, context_tokens, self.time_emb(t))

    @staticmethod
    def _t_broadcast(t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Reshape t to (B, T, 1) so it broadcasts over the dof axis.

        Supports a shared per-sequence level t of shape (B,) and a
        diffusion-forcing per-frame level t of shape (B, T).
        """
        if t.ndim == 1:
            return t.view(batch, 1, 1)
        if t.ndim == 2:
            return t.view(batch, seq_len, 1)
        raise ValueError(f"t must be rank-1 (B,) or rank-2 (B, T), got {tuple(t.shape)}.")

    def _time_tokens(self, t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Time embedding as (B, T, hidden), one per body frame.

        A shared t (B,) yields one embedding broadcast across frames; a
        per-frame t (B, T) yields a distinct embedding per frame.
        """
        emb = self.time_emb(t)
        if t.ndim == 1:
            return emb.unsqueeze(1).expand(batch, seq_len, emb.shape[-1])
        return emb

    def sample_training_tuple(
        self,
        s_full: torch.Tensor,
        t: torch.Tensor | None = None,
        use_logit_normal_t: bool = False,
        logit_normal_sigma: float = 1.0,
        per_frame: bool = False,
        obs_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a flow-matching training tuple.

        With ``per_frame=False`` every frame shares one noise level t (B,).
        With ``per_frame=True`` each frame gets an independent level t (B, T)
        (diffusion forcing). The clean-history / noisy-future rollout state
        itself is produced by obs_mask pinning (x_in re-imposes known_full
        regardless of t); per-frame t exists so the noise level can be a
        TRUTHFUL per-frame signal: pass ``obs_mask`` and fully-observed frames
        get t forced to exactly 1.0 ("this frame is clean"), matching what
        ``sample()`` feeds per_frame_noise checkpoints at inference. Without
        ``obs_mask`` the level on observed frames stays random and carries no
        information, so the model learns to ignore it — the sampler's t=1
        branch is then out of distribution.
        """
        batch, seq_len, _ = s_full.shape
        x_0 = torch.randn_like(s_full)
        if t is None:
            shape = (batch, seq_len) if per_frame else (batch,)
            if use_logit_normal_t:
                u = torch.randn(*shape, device=s_full.device, dtype=s_full.dtype) * logit_normal_sigma
                t = torch.sigmoid(u)
            else:
                t = torch.rand(*shape, device=s_full.device, dtype=s_full.dtype)
        if per_frame and obs_mask is not None and t.ndim == 2:
            # Truthful levels: fully-observed frames are clean by construction
            # (obs_mask pinning), so declare exactly t=1 — the same criterion
            # and value ``sample()`` uses. Partially pinned frames (EE-state
            # columns) keep their sampled level; per-frame t cannot express
            # per-column cleanliness.
            fully_obs = obs_mask.amin(dim=-1) >= 1.0
            t = t.masked_fill(fully_obs, 1.0)
        t_view = self._t_broadcast(t, batch, seq_len)
        x_t = (1.0 - t_view) * x_0 + t_view * s_full
        v_gt = s_full - x_0
        return x_t, v_gt, t, x_0

    def forward(
        self,
        x_t: torch.Tensor,
        known_full: torch.Tensor,
        obs_mask: torch.Tensor,
        s_ee: torch.Tensor,
        t: torch.Tensor,
        s_ee_look: torch.Tensor | None = None,
        look_valid: torch.Tensor | None = None,
    ) -> MaskedFlowTransformerOutput:
        batch, seq_len, _ = x_t.shape
        self._validate_condition(s_ee)
        x_in = self._assemble_observed_state(x_t, known_full, obs_mask)

        body_tokens = (
            self.feat_proj(x_in)
            # Frame-aligned EE: s_ee[:, t] goes onto the body token at the same t,
            # so frame t's hand condition is already IN frame t's input rather
            # than something attention must retrieve from the parallel ee_tokens.
            + self.ee_frame_proj(s_ee)
            + self.mask_proj(obs_mask)
            + self.pos_emb[:, :seq_len]
        )
        if self.temporal_backbone != "dit":
            body_tokens = body_tokens + self._time_tokens(t, batch, seq_len)
        ee_tokens = self.ee_proj(s_ee) + self.ee_pos_emb[:, :seq_len]
        tokens = [ee_tokens]
        if self.n_lookahead_tokens > 0:
            tokens.append(self._lookahead_tokens(s_ee, s_ee_look, look_valid))
        hidden = self._encode_body_tokens(
            body_tokens,
            torch.cat(tokens, dim=1),
            t,
        )

        if self.temporal_backbone != "dit":
            hidden = self.norm(hidden)
        pred_v = self.out_proj(hidden) * (1.0 - obs_mask)

        t_view = self._t_broadcast(t, batch, seq_len)
        denom = torch.clamp(1.0 - t_view, min=1e-4)
        pred_x1 = x_in + denom * pred_v
        pred_x1 = self._assemble_observed_state(pred_x1, known_full, obs_mask)

        return MaskedFlowTransformerOutput(pred_v=pred_v, pred_x1=pred_x1, x_in=x_in, t=t)

    @torch.no_grad()
    def sample(
        self,
        s_ee: torch.Tensor,
        known_full: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        s_ee_look: torch.Tensor | None = None,
        look_valid: torch.Tensor | None = None,
        rolling: float = 0.0,
    ) -> torch.Tensor:
        rolling = max(float(rolling), 0.0)
        if rolling > 0.0 and not bool(getattr(self.cfg, "per_frame_noise", False)):
            raise ValueError(
                "rolling schedule requires a per_frame_noise checkpoint; "
                "shared-t models never saw mixed per-frame noise levels."
            )
        was_training = self.training
        self.eval()
        self._validate_condition(s_ee)
        steps = num_steps or self.cfg.ode_steps
        batch, seq_len, _ = s_ee.shape
        n_dof = self.cfg.n_total_dof
        if known_full is None:
            known_full = torch.zeros(batch, seq_len, n_dof, device=s_ee.device, dtype=s_ee.dtype)
        if obs_mask is None:
            obs_mask = torch.zeros_like(known_full)
        ee_state_dim = int(getattr(self.cfg, "ee_state_dim", 0))
        if ee_state_dim > 0 and obs_mask[..., -ee_state_dim:].amin() < 1.0:
            # EE-as-state checkpoints are trained with the trailing EE columns
            # always observed; sampling without them is out of distribution.
            known_full = known_full.clone()
            obs_mask = obs_mask.clone()
            apply_ee_state_condition(known_full, obs_mask, s_ee[..., :ee_state_dim])
        x = torch.randn(batch, seq_len, n_dof, device=s_ee.device, dtype=s_ee.dtype) * temperature
        x = self._assemble_observed_state(x, known_full, obs_mask)
        dt = 1.0 / max(steps, 1)

        # Diffusion-forcing checkpoints (per_frame_noise) are trained with
        # truthful per-frame levels: fully-observed frames carry exactly t=1
        # (forced in sample_training_tuple via obs_mask) while other frames
        # get independent random levels. Sampling mirrors that contract:
        # fully-observed frames say t=1, generated frames sweep 0->1.
        # Partially pinned frames (EE-state columns) keep the sweep level --
        # t is per-frame and cannot express per-column cleanliness.
        # Shared-t checkpoints never saw mixed per-frame levels; a per-frame t
        # would be out of distribution for them, so they keep the scalar t.
        #
        # ``rolling`` > 0 staggers the sweep by temporal position (diffusion
        # forcing rolling schedule): frame with rank r sweeps
        # t(k) = clamp(base*(1+rolling) - rolling*r, 0, 1), so early frames
        # commit first and the still-noisy tail is denoised against an
        # already-stable head. Training sampled INDEPENDENT per-frame levels,
        # so every such monotone combination is in-distribution. Each frame's
        # own sweep compresses to 1/(1+rolling) of the steps -- scale
        # ``num_steps`` up accordingly. rolling=0 reproduces the old sweep.
        per_frame_cfg = bool(getattr(self.cfg, "per_frame_noise", False))
        fully_obs = obs_mask.amin(dim=-1) >= 1.0 if per_frame_cfg else None
        per_frame_t = per_frame_cfg and (rolling > 0.0 or bool(fully_obs.any()))

        rank = None
        if rolling > 0.0:
            # Temporal rank in [0, 1] over the generated frames of each row.
            gen = (~fully_obs).to(x.dtype)
            order = torch.cumsum(gen, dim=1) - 1.0
            span = (gen.sum(dim=1, keepdim=True) - 1.0).clamp_min(1.0)
            rank = (order / span).clamp(0.0, 1.0) * gen

        def _schedule(base: float) -> torch.Tensor:
            if rank is not None:
                t = (base * (1.0 + rolling) - rolling * rank).clamp(0.0, 1.0)
            else:
                t = torch.full(
                    (x.shape[0], x.shape[1]), base, device=x.device, dtype=x.dtype
                )
            return t.masked_fill(fully_obs, 1.0)

        for step in range(steps):
            base = step / max(steps, 1)
            base_next = min(base + dt, 1.0)
            if per_frame_t:
                t = _schedule(base)
                t_view: torch.Tensor | float = t.unsqueeze(-1)
                t_next_view: torch.Tensor | float = _schedule(base_next).unsqueeze(-1)
                denom: torch.Tensor | float = (1.0 - t_view).clamp_min(1e-4)
            else:
                t = torch.full((x.shape[0],), base, device=x.device, dtype=x.dtype)
                t_view = base
                t_next_view = base_next
                denom = max(1.0 - base, 1e-4)
            out = self.forward(
                x, known_full, obs_mask, s_ee, t,
                s_ee_look=s_ee_look, look_valid=look_valid,
            )
            x_0_est = (x - t_view * out.pred_x1) / denom
            x = (1.0 - t_next_view) * x_0_est + t_next_view * out.pred_x1
            x = self._assemble_observed_state(x, known_full, obs_mask)

        self.train(was_training)
        return x

    def n_params(self) -> dict[str, int]:
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        backbone = self.transformer
        if self.temporal_backbone == "hierarchical":
            backbone = self.temporal_pyramid
        elif self.temporal_backbone == "dit":
            backbone = self.temporal_dit
        if backbone is None:
            raise TemporalPyramidConfigError(
                "temporal_backbone",
                self.temporal_backbone,
            )
        return {"transformer": _count(backbone), "total": _count(self)}


@torch.no_grad()
def sample_autoregressive_primitives(
    model: EEMaskedFlowTransformer,
    s_ee_prim: torch.Tensor,
    initial_history: torch.Tensor,
    num_steps: int | None = None,
    temperature: float = 1.0,
    s_ee_look_prim: torch.Tensor | None = None,
    look_valid_prim: torch.Tensor | None = None,
    rolling: float = 0.0,
) -> torch.Tensor:
    """Generate a full segment by rolling primitive predictions forward."""
    prim_cfg = getattr(model.cfg, "primitive", PrimitiveConfig())
    if not getattr(prim_cfg, "enabled", False):
        raise ValueError("Autoregressive primitive sampling requires cfg.primitive.enabled=True.")

    squeeze_batch = False
    if s_ee_prim.ndim == 3:
        s_ee_prim = s_ee_prim.unsqueeze(0)
        squeeze_batch = True
    if initial_history.ndim == 2:
        initial_history = initial_history.unsqueeze(0)
    if s_ee_look_prim is not None and s_ee_look_prim.ndim == 3:
        s_ee_look_prim = s_ee_look_prim.unsqueeze(0)
    if look_valid_prim is not None and look_valid_prim.ndim == 2:
        look_valid_prim = look_valid_prim.unsqueeze(0)
    if s_ee_prim.ndim != 4:
        raise ValueError(
            f"s_ee_prim must be rank-4 (B, N, T, E) or rank-3 (N, T, E), got {tuple(s_ee_prim.shape)}."
        )
    if initial_history.ndim != 3:
        raise ValueError(
            f"initial_history must be rank-3 (B, H, D) or rank-2 (H, D), got {tuple(initial_history.shape)}."
        )

    batch, num_primitives, seq_len, _ = s_ee_prim.shape
    history_len = initial_history.shape[1]
    if batch != initial_history.shape[0]:
        raise ValueError(
            f"Batch mismatch between s_ee_prim {tuple(s_ee_prim.shape)} and initial_history {tuple(initial_history.shape)}."
        )
    if seq_len != prim_cfg.primitive_len:
        raise ValueError(
            f"Expected primitive length {prim_cfg.primitive_len}, got {seq_len}."
        )
    if history_len != prim_cfg.history_len:
        raise ValueError(
            f"Expected history length {prim_cfg.history_len}, got {history_len}."
        )

    history = initial_history
    generated = [history]
    for prim_idx in range(num_primitives):
        known_full, obs_mask = build_prefix_condition_tensors(history, seq_len)
        pred = model.sample(
            s_ee_prim[:, prim_idx],
            known_full=known_full,
            obs_mask=obs_mask,
            num_steps=num_steps,
            temperature=temperature,
            s_ee_look=(
                s_ee_look_prim[:, prim_idx] if s_ee_look_prim is not None else None
            ),
            look_valid=(
                look_valid_prim[:, prim_idx] if look_valid_prim is not None else None
            ),
            rolling=rolling,
        )
        pred[..., 5:7] = project_contact_probability(pred[..., 5:7])
        future = pred[:, history_len:]
        generated.append(future)
        history = pred[:, -history_len:]

    full_segment = torch.cat(generated, dim=1)
    return full_segment.squeeze(0) if squeeze_batch else full_segment


def save_masked_flow_checkpoint(
    model: EEMaskedFlowTransformer,
    ema_model: EEMaskedFlowTransformer,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    best_val: float,
    cfg,
    path: str | Path,
    extra_state: dict | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "cfg": cfg,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def load_masked_flow_state_dict(
    model: EEMaskedFlowTransformer,
    state_dict: dict[str, torch.Tensor],
) -> None:
    state_dict = dict(state_dict)
    if any(
        key.startswith("ee_cross_attn") or key.startswith("ee_cross_attn_norm")
        for key in state_dict
    ):
        raise RuntimeError(
            "Cross-attention masked-flow checkpoints are no longer supported. "
            "Use a prefix-architecture checkpoint instead."
        )
    if any(key.startswith("body_proj") for key in state_dict):
        raise RuntimeError(
            "Legacy additive masked-flow checkpoints are no longer supported. "
            "Use a prefix-architecture checkpoint instead."
        )

    # Older checkpoints stored these as non-trainable buffers. They were
    # removed from the current model because delta69 no longer uses them.
    state_dict.pop("ee_mask", None)
    state_dict.pop("body_mask", None)

    # Frame-aligned EE injection (ee_frame_proj) was added after some
    # checkpoints were trained. When a checkpoint predates it, inject
    # zero-filled weights so ee_frame_proj(s_ee) == 0 for every input: the
    # body-token sum in forward() then reproduces the pre-ee_frame_proj
    # architecture exactly, so the older checkpoint runs unchanged.
    if hasattr(model, "ee_frame_proj") and not any(
        key.startswith("ee_frame_proj") for key in state_dict
    ):
        for name, param in model.ee_frame_proj.state_dict().items():
            state_dict[f"ee_frame_proj.{name}"] = torch.zeros_like(param)

    model.load_state_dict(state_dict)


def load_masked_flow_from_checkpoint(
    ckpt: str | Path | dict,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> tuple[EEMaskedFlowTransformer, object, dict]:
    if isinstance(ckpt, (str, Path)):
        state = torch.load(ckpt, map_location=device, weights_only=False)
    else:
        state = ckpt

    cfg = state["cfg"]
    if not hasattr(cfg, "primitive"):
        cfg.primitive = PrimitiveConfig()
    sd = state.get("ema_model", state["model"]) if use_ema else state["model"]

    if "transformer.layers.0.linear1.weight" in sd:
        linear1_weight = sd["transformer.layers.0.linear1.weight"]
        cfg.transformer_ffn_mult = linear1_weight.shape[0] // cfg.hidden_dim
    elif "temporal_pyramid.fine_encoder.layers.0.linear1.weight" in sd:
        linear1_weight = sd[
            "temporal_pyramid.fine_encoder.layers.0.linear1.weight"
        ]
        cfg.transformer_ffn_mult = linear1_weight.shape[0] // cfg.hidden_dim
    elif "temporal_dit.blocks.0.mlp.0.weight" in sd:
        linear1_weight = sd["temporal_dit.blocks.0.mlp.0.weight"]
        cfg.transformer_ffn_mult = linear1_weight.shape[0] // cfg.hidden_dim
    else:
        cfg.transformer_ffn_mult = 8

    if "out_proj.weight" in sd:
        cfg.out_proj_hidden_mult = 1
    elif "out_proj.0.weight" in sd and "out_proj.2.weight" in sd:
        cfg.out_proj_hidden_mult = sd["out_proj.0.weight"].shape[0] // cfg.hidden_dim
    else:
        cfg.out_proj_hidden_mult = 2

    model = EEMaskedFlowTransformer(cfg).to(device)
    load_masked_flow_state_dict(model, sd)
    return model, cfg, state

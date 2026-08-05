"""Two-stage (root -> body) cascade over the masked-flow motion prior.

Stage 1 (root model) is a second, smaller EEMaskedFlowTransformer over the
ROOT-CHANNEL SUBSET of the delta69/73 features:

    full indices [0:11]  : tilt(4) + delta_yaw(1) + contact(2) + delta_trans(3) + height(1)
    full indices [69:73] : xy_rel(2) + yaw_rel cos/sin(2)   (abs_root_channels only)

Cascade order is SEQUENTIAL AT THE SEGMENT LEVEL: stage 1 first rolls out the
root for the WHOLE segment (2 history + 32 future frames, via the standard
4x8 primitive rollout), and only then does stage 2 generate the body. This
ordering exists because the body stage must see the FULL 34-frame root
trajectory, not just its own 10-frame window:

  * in-window (10 frames)  : root columns of known_full/obs_mask pinned to the
    stage-1 values on the generated rows (the pinned history prefix keeps its
    own root columns) -- the same mechanism as the EE-state columns;
  * beyond-window (24 frames): stage-1 root fed as soft PREVIEW TOKENS,
    downsampled by ``root_look_stride`` -- the same convention as the EE
    lookahead (hold-pad, prefix validity, learned invalid embedding), so the
    legs can phase their gait against where the root is going, not just where
    it is. A per-denoising-step cascade cannot provide this: the future
    windows' root does not exist yet inside a step.

Stage-1 outputs are DETACHED before stage 2 consumes them (pin + preview):
the body loss must not drag the root toward "whatever makes the body easy".
Both stages share the s_ee condition window and EE lookahead preview, so the
primitive/lookahead machinery is reused as-is.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from Prior_Recon.Masked_Flow.config import PrimitiveConfig
from Prior_Recon.Masked_Flow.model.masked_flow_transformer import (
    EEMaskedFlowTransformer,
    MaskedFlowTransformerOutput,
    apply_ee_state_condition,
    build_prefix_condition_tensors,
    load_masked_flow_state_dict,
    sample_autoregressive_primitives,
)
from Prior_Recon.Masked_Flow.model.root_model_config import (
    derive_root_model_cfg,
)

ROOT_STATE_DIMS = 11  # tilt(4) + delta_yaw(1) + contact(2) + delta_trans(3) + height(1)
ABS_ROOT_DIMS = 4  # xy_rel(2) + yaw_rel cos/sin(2)
# Local (root-subset) indices of the per-frame DELTA channels: delta_yaw and
# delta_trans. Preview padding zeroes them (a held pose has zero rates), the
# same convention as the EE lookahead's zeroed velocity block.
_ROOT_DELTA_LOCAL_DIMS = (4, 7, 8, 9)


def root_channel_indices(cfg) -> list[int]:
    """Full-feature indices of the root+contact channels stage 1 predicts."""
    idx = list(range(ROOT_STATE_DIMS))
    if getattr(cfg, "abs_root_channels", False):
        n_body = int(cfg.n_motion_dof)
        idx += list(range(n_body - ABS_ROOT_DIMS, n_body))
    return idx


class BodyStageTransformer(EEMaskedFlowTransformer):
    """Body model + a root-preview token stream (stage-1 root beyond the window).

    Same feature format as the pinned root columns, distinguished by its own
    positional slots and projection; validity semantics mirror the EE
    lookahead (prefix-valid, hold-pad, zero-init invalid embedding, missing
    preview degrades to all-invalid).
    """

    def __init__(self, cfg, root_dim: int):
        super().__init__(cfg)
        self.root_dim = int(root_dim)
        self.root_look_len = int(getattr(cfg, "root_look_len", 0))
        self.root_look_stride = max(int(getattr(cfg, "root_look_stride", 1)), 1)
        self.n_root_look_tokens = int(getattr(cfg, "n_root_look_tokens", 0))
        if self.n_root_look_tokens > 0:
            hidden = cfg.hidden_dim
            self.root_look_proj = nn.Sequential(
                nn.Linear(self.root_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.root_look_pos_emb = nn.Parameter(
                torch.zeros(1, self.n_root_look_tokens, hidden)
            )
            nn.init.trunc_normal_(self.root_look_pos_emb, std=0.02)
            self.root_look_invalid_emb = nn.Parameter(torch.zeros(1, 1, hidden))

    def _root_lookahead_tokens(
        self,
        root_look: torch.Tensor | None,
        root_look_valid: torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if root_look is None:
            root_look = torch.zeros(
                batch, self.root_look_len, self.root_dim, device=device, dtype=dtype
            )
            root_look_valid = None
        if root_look.shape[1] != self.root_look_len or root_look.shape[-1] != self.root_dim:
            raise ValueError(
                f"root_look must be (B, {self.root_look_len}, {self.root_dim}), "
                f"got {tuple(root_look.shape)}."
            )
        if root_look_valid is None:
            root_look_valid = torch.zeros(
                batch, self.root_look_len, device=device, dtype=dtype
            )
        if root_look_valid.shape[:2] != root_look.shape[:2]:
            raise ValueError(
                f"root_look_valid shape {tuple(root_look_valid.shape)} must match "
                f"root_look {tuple(root_look.shape)} on (B, K)."
            )
        look_ds = root_look[:, :: self.root_look_stride][:, : self.n_root_look_tokens]
        valid_ds = root_look_valid[:, :: self.root_look_stride][:, : self.n_root_look_tokens]
        tokens = self.root_look_proj(look_ds) + self.root_look_pos_emb
        tokens = tokens + (1.0 - valid_ds.unsqueeze(-1)) * self.root_look_invalid_emb
        return tokens

    def forward(  # noqa: D102 -- parent forward + the root preview stream
        self,
        x_t: torch.Tensor,
        known_full: torch.Tensor,
        obs_mask: torch.Tensor,
        s_ee: torch.Tensor,
        t: torch.Tensor,
        s_ee_look: torch.Tensor | None = None,
        look_valid: torch.Tensor | None = None,
        root_look: torch.Tensor | None = None,
        root_look_valid: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = x_t.shape
        self._validate_condition(s_ee)
        x_in = self._assemble_observed_state(x_t, known_full, obs_mask)

        body_tokens = (
            self.feat_proj(x_in)
            + self.mask_proj(obs_mask)
            + self.pos_emb[:, :seq_len]
        )
        if self.temporal_backbone != "dit":
            body_tokens = body_tokens + self._time_tokens(t, batch, seq_len)
        ee_tokens = self.ee_proj(s_ee) + self.ee_pos_emb[:, :seq_len]
        tokens = [ee_tokens]
        context_key_padding_mask = None
        if self.n_lookahead_tokens > 0:
            look_tokens, look_valid_ds = self._lookahead_tokens(
                s_ee,
                s_ee_look,
                look_valid,
            )
            tokens.append(look_tokens)
            if self.mask_invalid_lookahead:
                context_key_padding_mask = self._context_key_padding_mask(
                    look_valid_ds,
                    seq_len,
                )
        if self.n_root_look_tokens > 0:
            root_look_tokens = self._root_lookahead_tokens(
                root_look, root_look_valid, batch, x_t.device, x_t.dtype
            )
            tokens.append(root_look_tokens)
            if context_key_padding_mask is not None:
                # The root preview stream is appended AFTER the EE preview, so
                # the mask built for [window | ee preview] must be extended or
                # it would misalign with the key set.
                context_key_padding_mask = torch.cat(
                    [
                        context_key_padding_mask,
                        torch.zeros(
                            batch,
                            root_look_tokens.shape[1],
                            device=context_key_padding_mask.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )
        hidden = self._encode_body_tokens(
            body_tokens,
            torch.cat(tokens, dim=1),
            t,
            context_key_padding_mask,
        )

        if self.norm is not None:
            hidden = self.norm(hidden)
        pred_v = self.out_proj(hidden) * (1.0 - obs_mask)

        t_view = self._t_broadcast(t, batch, seq_len)
        denom = torch.clamp(1.0 - t_view, min=1e-4)
        pred_x1 = x_in + denom * pred_v
        pred_x1 = self._assemble_observed_state(pred_x1, known_full, obs_mask)
        return MaskedFlowTransformerOutput(pred_v=pred_v, pred_x1=pred_x1, x_in=x_in, t=t)


class TwoStageMaskedFlow(nn.Module):
    """root_model (segment rollout first) + body_model (pin + root preview)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        idx = root_channel_indices(cfg)
        n_body = int(cfg.n_motion_dof)
        if max(idx) >= n_body:
            raise ValueError(f"Root indices {idx} exceed motion feature width {n_body}.")
        self.register_buffer(
            "root_idx", torch.tensor(idx, dtype=torch.long), persistent=False
        )
        root_cfg = derive_root_model_cfg(cfg, len(root_channel_indices(cfg)))
        self.root_model = EEMaskedFlowTransformer(root_cfg)
        self.body_model = BodyStageTransformer(cfg, root_dim=len(idx))

    # ── channel plumbing ────────────────────────────────────────────────────
    def gather_root(self, x: torch.Tensor) -> torch.Tensor:
        """Root-channel subset of a full feature tensor (copy)."""
        return x.index_select(-1, self.root_idx)

    def pin_window_root(
        self,
        known_full: torch.Tensor,
        obs_mask: torch.Tensor,
        window_root: torch.Tensor,
    ) -> None:
        """Pin the stage-1 root into the body condition (in place).

        Only rows that are NOT already fully observed take the stage-1 values;
        the pinned history prefix keeps its own root columns (at inference the
        two coincide -- the body history's root columns were themselves pinned
        to the stage-1 chain -- but under teacher forcing the GT history must
        stay intact).
        """
        if window_root.shape[-1] != self.root_idx.numel():
            raise ValueError(
                f"window_root last dim {window_root.shape[-1]} != "
                f"{self.root_idx.numel()} root channels."
            )
        row_full = (obs_mask.amin(dim=-1) >= 1.0).unsqueeze(-1)  # (B, T, 1)
        known_full[..., self.root_idx] = torch.where(
            row_full, known_full[..., self.root_idx], window_root
        )
        obs_mask[..., self.root_idx] = torch.where(
            row_full,
            obs_mask[..., self.root_idx],
            torch.ones_like(window_root),
        )

    def build_root_preview(
        self,
        segment_root: torch.Tensor,
        prim_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage-1 root beyond the window as (preview, validity), hold-padded.

        ``segment_root`` is the full stage-1 segment estimate (B, S, R);
        the window is [prim_start, prim_start + primitive_len). The preview is
        padded to root_look_len frames by holding the last available frame with
        the delta channels zeroed (held pose => zero rates), validity is
        prefix-form -- identical semantics to the dataset's EE lookahead, and
        this ONE builder serves training and sampling so the two match.
        """
        prim_len = int(self.cfg.primitive.primitive_len)
        k = int(getattr(self.cfg, "root_look_len", 0))
        batch, seg_len, root_dim = segment_root.shape
        end = prim_start + prim_len
        if end > seg_len:
            raise ValueError(f"Window end {end} exceeds segment length {seg_len}.")
        avail = segment_root[:, end:]
        n_real = min(avail.shape[1], k)
        valid = torch.zeros(batch, k, device=segment_root.device, dtype=segment_root.dtype)
        valid[:, :n_real] = 1.0
        hold = (avail[:, n_real - 1 : n_real] if n_real > 0 else segment_root[:, end - 1 : end]).clone()
        hold[..., list(_ROOT_DELTA_LOCAL_DIMS)] = 0.0
        preview = torch.cat(
            [avail[:, :n_real], hold.expand(batch, k - n_real, root_dim)], dim=1
        )
        return preview, valid

    def n_params(self) -> dict[str, int]:
        root = self.root_model.n_params()
        body = self.body_model.n_params()
        return {
            "transformer": root["transformer"] + body["transformer"],
            "root_total": root["total"],
            "body_total": body["total"],
            "total": root["total"] + body["total"],
        }

    # ── sampling ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_body_window(
        self,
        s_ee: torch.Tensor,
        window_root: torch.Tensor,
        known_full: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        s_ee_look: torch.Tensor | None = None,
        look_valid: torch.Tensor | None = None,
        root_look: torch.Tensor | None = None,
        root_look_valid: torch.Tensor | None = None,
        rolling: float = 0.0,
    ) -> torch.Tensor:
        """One body window given the stage-1 root (in-window pin + preview).

        Mirrors EEMaskedFlowTransformer.sample (per-frame truthful-t contract,
        rolling schedule) with the root condition added.
        """
        rolling = max(float(rolling), 0.0)
        per_frame_cfg = bool(getattr(self.cfg, "per_frame_noise", False))
        if rolling > 0.0 and not per_frame_cfg:
            raise ValueError(
                "rolling schedule requires a per_frame_noise checkpoint; "
                "shared-t models never saw mixed per-frame noise levels."
            )
        was_training = self.training
        self.eval()

        steps = num_steps or self.cfg.ode_steps
        batch, seq_len, _ = s_ee.shape
        n_dof = self.cfg.n_total_dof
        device, dtype = s_ee.device, s_ee.dtype
        if known_full is None:
            known_full = torch.zeros(batch, seq_len, n_dof, device=device, dtype=dtype)
        if obs_mask is None:
            obs_mask = torch.zeros_like(known_full)
        known_full = known_full.clone()
        obs_mask = obs_mask.clone()
        ee_state_dim = int(getattr(self.cfg, "ee_state_dim", 0))
        if ee_state_dim > 0 and obs_mask[..., -ee_state_dim:].amin() < 1.0:
            apply_ee_state_condition(known_full, obs_mask, s_ee[..., :ee_state_dim])

        # Truthful-t rows are judged BEFORE the root pin (the criterion training
        # uses); pinning root columns cannot complete a partially observed row.
        fully_obs = obs_mask.amin(dim=-1) >= 1.0 if per_frame_cfg else None
        self.pin_window_root(known_full, obs_mask, window_root)

        x = torch.randn(batch, seq_len, n_dof, device=device, dtype=dtype) * temperature
        x = self.body_model._assemble_observed_state(x, known_full, obs_mask)
        dt = 1.0 / max(steps, 1)
        per_frame_t = per_frame_cfg and (rolling > 0.0 or bool(fully_obs.any()))

        rank = None
        if rolling > 0.0:
            gen = (~fully_obs).to(dtype)
            order = torch.cumsum(gen, dim=1) - 1.0
            span = (gen.sum(dim=1, keepdim=True) - 1.0).clamp_min(1.0)
            rank = (order / span).clamp(0.0, 1.0) * gen

        def _schedule(base: float) -> torch.Tensor:
            if rank is not None:
                t = (base * (1.0 + rolling) - rolling * rank).clamp(0.0, 1.0)
            else:
                t = torch.full((batch, seq_len), base, device=device, dtype=dtype)
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
                t = torch.full((batch,), base, device=device, dtype=dtype)
                t_view = base
                t_next_view = base_next
                denom = max(1.0 - base, 1e-4)
            out = self.body_model(
                x, known_full, obs_mask, s_ee, t,
                s_ee_look=s_ee_look, look_valid=look_valid,
                root_look=root_look, root_look_valid=root_look_valid,
            )
            x0_est = (x - t_view * out.pred_x1) / denom
            x = (1.0 - t_next_view) * x0_est + t_next_view * out.pred_x1
            x = self.body_model._assemble_observed_state(x, known_full, obs_mask)

        self.train(was_training)
        return x

    @torch.no_grad()
    def sample_segment(
        self,
        s_ee_prim: torch.Tensor,
        initial_history: torch.Tensor,
        num_steps: int | None = None,
        temperature: float = 1.0,
        s_ee_look_prim: torch.Tensor | None = None,
        look_valid_prim: torch.Tensor | None = None,
        rolling: float = 0.0,
    ) -> torch.Tensor:
        """Full-segment generation: stage-1 root rollout FIRST, then the body.

        ``initial_history`` is FULL-dim (B, H, D) or (H, D); the root history
        is its gathered root columns.
        """
        prim_cfg = getattr(self.cfg, "primitive", PrimitiveConfig())
        if not getattr(prim_cfg, "enabled", False):
            raise ValueError("Segment sampling requires cfg.primitive.enabled=True.")

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
                f"s_ee_prim must be rank-4 (B, N, T, E) or rank-3 (N, T, E), "
                f"got {tuple(s_ee_prim.shape)}."
            )
        if initial_history.ndim != 3:
            raise ValueError(
                f"initial_history must be rank-3 (B, H, D) or rank-2 (H, D), "
                f"got {tuple(initial_history.shape)}."
            )
        batch, num_primitives, seq_len, _ = s_ee_prim.shape
        history_len = initial_history.shape[1]
        if batch != initial_history.shape[0]:
            raise ValueError(
                f"Batch mismatch between s_ee_prim {tuple(s_ee_prim.shape)} and "
                f"initial_history {tuple(initial_history.shape)}."
            )
        if seq_len != prim_cfg.primitive_len:
            raise ValueError(f"Expected primitive length {prim_cfg.primitive_len}, got {seq_len}.")
        if history_len != prim_cfg.history_len:
            raise ValueError(f"Expected history length {prim_cfg.history_len}, got {history_len}.")

        # ── stage 1: root rollout over the whole segment ─────────────────────
        segment_root = sample_autoregressive_primitives(
            self.root_model,
            s_ee_prim,
            self.gather_root(initial_history),
            num_steps=num_steps,
            temperature=temperature,
            s_ee_look_prim=s_ee_look_prim,
            look_valid_prim=look_valid_prim,
            rolling=rolling,
        )  # (B, segment_len, R)

        # ── stage 2: body rollout with in-window pin + root preview ─────────
        history = initial_history
        generated = [history]
        for prim_idx in range(num_primitives):
            prim_start = prim_idx * prim_cfg.future_len
            known_full, obs_mask = build_prefix_condition_tensors(history, seq_len)
            window_root = segment_root[:, prim_start : prim_start + seq_len]
            root_look, root_look_valid = self.build_root_preview(segment_root, prim_start)
            pred = self.sample_body_window(
                s_ee_prim[:, prim_idx],
                window_root,
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
                root_look=root_look,
                root_look_valid=root_look_valid,
                rolling=rolling,
            )
            generated.append(pred[:, history_len:])
            history = pred[:, -history_len:]

        full_segment = torch.cat(generated, dim=1)
        return full_segment.squeeze(0) if squeeze_batch else full_segment


def sample_autoregressive_primitives_two_stage(
    model: TwoStageMaskedFlow,
    s_ee_prim: torch.Tensor,
    initial_history: torch.Tensor,
    num_steps: int | None = None,
    temperature: float = 1.0,
    s_ee_look_prim: torch.Tensor | None = None,
    look_valid_prim: torch.Tensor | None = None,
    rolling: float = 0.0,
) -> torch.Tensor:
    """API-symmetric wrapper around TwoStageMaskedFlow.sample_segment."""
    return model.sample_segment(
        s_ee_prim,
        initial_history,
        num_steps=num_steps,
        temperature=temperature,
        s_ee_look_prim=s_ee_look_prim,
        look_valid_prim=look_valid_prim,
        rolling=rolling,
    )


def load_two_stage_from_checkpoint(
    ckpt: str | Path | dict,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> tuple[TwoStageMaskedFlow, object, dict]:
    """Load a cascade checkpoint saved by the two-stage trainer."""
    if isinstance(ckpt, (str, Path)):
        state = torch.load(ckpt, map_location=device, weights_only=False)
    else:
        state = ckpt

    cfg = state["cfg"]
    if not hasattr(cfg, "root_stage"):
        raise ValueError(
            "Checkpoint cfg has no root_stage block -- this is a single-stage "
            "checkpoint; use load_masked_flow_from_checkpoint instead."
        )
    sd = state.get("ema_model", state["model"]) if use_ema else state["model"]
    root_dit_key = "root_model.temporal_dit.blocks.0.mlp.0.weight"
    if root_dit_key in sd:
        cfg.root_stage.transformer_ffn_mult = (
            sd[root_dit_key].shape[0] // cfg.root_stage.hidden_dim
        )
    body_dit_key = "body_model.temporal_dit.blocks.0.mlp.0.weight"
    if body_dit_key in sd:
        cfg.transformer_ffn_mult = (
            sd[body_dit_key].shape[0] // cfg.hidden_dim
        )
    model = TwoStageMaskedFlow(cfg).to(device)
    load_masked_flow_state_dict(model, sd)
    return model, cfg, state

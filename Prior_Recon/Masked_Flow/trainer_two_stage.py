"""Trainer for the two-stage (root -> body) cascade masked-flow prior.

Subclasses MaskedFlowTransformerTrainer so all shared infrastructure (DDP,
EMA, microbatch OOM guard, curriculum progress, checkpointing, rollout
schedule, history perturbation) is inherited unchanged.

Each training step runs TWO passes over the segment's primitives, mirroring
TwoStageMaskedFlow.sample_segment:

  pass 1 (root)  : per primitive, one root forward at a random noise level,
                   history rolled out from the root model's own predictions;
                   the per-primitive pred_x1 futures are stitched (detached)
                   into a full 34-frame segment root estimate.
  pass 2 (body)  : per primitive, the body model sees the FULL segment root --
                   in-window frames via pinned root columns, beyond-window
                   frames via root preview tokens -- both from the stage-1
                   estimate, both detached.

Joint total = body.total + root_stage.stage_weight * root.total, one backward.
The detach guarantees the body loss contributes no gradient to the root model.
"""
from __future__ import annotations

import torch

from Prior_Recon.Masked_Flow.loss.masked_flow_loss import (
    MaskedFlowLossOutput,
    mean_loss_outputs,
    merge_segment_geometry,
)
from Prior_Recon.Masked_Flow.loss.root_stage_loss import (
    RootStageLoss,
    RootStageLossOutput,
    mean_root_loss_outputs,
)
from Prior_Recon.Masked_Flow.model.masked_flow_transformer import (
    apply_ee_state_condition,
    build_prefix_condition_tensors,
)
from Prior_Recon.Masked_Flow.model.two_stage_cascade import TwoStageMaskedFlow
from Prior_Recon.Masked_Flow.trainer_masked_flow import (
    MaskedFlowTransformerTrainer,
)


class TwoStageLossValues:
    """Joint loss container with the duck-typed surface the base trainer uses.

    ``total`` backs backward(); ``as_dict`` feeds logging with both the body
    ("loss/*") and root ("root/*") families; "loss/total" is overridden with
    the JOINT total so best-checkpoint selection and epoch logs reflect what
    is actually optimized.
    """

    def __init__(
        self,
        body: MaskedFlowLossOutput,
        root: RootStageLossOutput,
        stage_weight: float,
    ):
        self.body = body
        self.root = root
        self.total = body.total + stage_weight * root.total

    def as_dict(
        self,
        include_quantize_rot: bool = True,
        include_quantize_trans: bool = True,
    ) -> dict[str, float]:
        values = self.body.as_dict(
            include_quantize_rot=include_quantize_rot,
            include_quantize_trans=include_quantize_trans,
        )
        values["loss/body_total"] = values["loss/total"]
        values.update(self.root.as_dict())
        values["loss/total"] = self.total.item()
        return values


class TwoStageMaskedFlowTrainer(MaskedFlowTransformerTrainer):
    def __init__(self, cfg, **kwargs):
        if not hasattr(cfg, "root_stage"):
            raise ValueError(
                "TwoStageMaskedFlowTrainer needs a TwoStageMaskedFlowConfig "
                "(cfg.root_stage missing) -- load it via two_stage_config_from_yaml."
            )
        super().__init__(cfg, **kwargs)
        self.root_loss_fn = RootStageLoss(cfg)

    def _build_model(self, cfg) -> TwoStageMaskedFlow:
        return TwoStageMaskedFlow(cfg)

    def _perturb_root_history(self, history_root: torch.Tensor, train: bool) -> torch.Tensor:
        """Layer B analog in the root subset: tilt-only perturbation.

        The base _perturb_history writes joint channels 11:40, which in the
        15-dim root layout would corrupt the abs-root block -- so the root pass
        re-implements just the tilt part (local dims 0:4, same encoding), with
        the same probability/std knobs. The root subset has no joint channels,
        so tilt-only IS the full analog.
        """
        prob = float(getattr(self.cfg, "history_perturb_prob", 0.0))
        t_std = float(getattr(self.cfg, "history_perturb_tilt_std", 0.0))
        if not train or prob <= 0.0 or t_std <= 0.0:
            return history_root
        batch = history_root.shape[0]
        dev, dt = history_root.device, history_root.dtype
        fire = (torch.rand(batch, 1, device=dev, dtype=dt) < prob).to(dt)
        out = history_root.clone()
        roll = torch.atan2(out[..., 0], out[..., 1] + 1.0)
        pitch = torch.atan2(out[..., 2], out[..., 3] + 1.0)
        roll = roll + torch.randn(batch, 1, device=dev, dtype=dt) * t_std * fire
        pitch = pitch + torch.randn(batch, 1, device=dev, dtype=dt) * t_std * fire
        out[..., 0] = torch.sin(roll)
        out[..., 1] = torch.cos(roll) - 1.0
        out[..., 2] = torch.sin(pitch)
        out[..., 3] = torch.cos(pitch) - 1.0
        return out

    def _step_primitives(
        self,
        batch: dict,
        train: bool,
        force_teacher: bool,
    ) -> TwoStageLossValues:
        s_full_prim = batch["s_full_prim"]
        s_ee_prim = batch["s_ee_prim"]
        s_ee_look_prim = batch.get("s_ee_look_prim")
        look_valid_prim = batch.get("look_valid_prim")
        segment_full = batch.get("segment_full")
        prim_cfg = self.primitive_cfg
        history_len = prim_cfg.history_len
        if s_full_prim.shape[1] != prim_cfg.num_primitives:
            raise ValueError(
                f"Expected {prim_cfg.num_primitives} primitives, got {s_full_prim.shape[1]}."
            )
        if s_full_prim.shape[2] != prim_cfg.primitive_len:
            raise ValueError(
                f"Expected primitive length {prim_cfg.primitive_len}, got {s_full_prim.shape[2]}."
            )

        use_lnt = getattr(self.cfg, "use_logit_normal_t", False)
        lnt_sigma = getattr(self.cfg, "logit_normal_sigma", 1.0)
        per_frame = getattr(self.cfg, "per_frame_noise", False)
        ee_state_dim = int(getattr(self.cfg, "ee_state_dim", 0))
        model: TwoStageMaskedFlow = self.model

        def _look(prim_idx: int):
            s_look = s_ee_look_prim[:, prim_idx] if s_ee_look_prim is not None else None
            l_valid = look_valid_prim[:, prim_idx] if look_valid_prim is not None else None
            return s_look, l_valid

        # ── pass 1: root over the whole segment ─────────────────────────────
        root_losses: list[RootStageLossOutput] = []
        prev_root_pred: torch.Tensor | None = None
        segment_root_parts: list[torch.Tensor] = []
        for prim_idx in range(prim_cfg.num_primitives):
            s_full_root = model.gather_root(s_full_prim[:, prim_idx])
            gt_history_root = s_full_root[:, :history_len]
            history_root = self._select_history(
                gt_history_root,
                prev_pred=prev_root_pred,
                train=train,
                force_teacher=(force_teacher or prim_idx == 0),
            )
            history_root = self._perturb_root_history(history_root, train=train)
            known_root, obs_root = build_prefix_condition_tensors(
                history_root, s_full_root.shape[1]
            )
            x_t_root, v_gt_root, t_root, _ = model.root_model.sample_training_tuple(
                s_full_root,
                use_logit_normal_t=use_lnt,
                logit_normal_sigma=lnt_sigma,
                per_frame=per_frame,
                obs_mask=obs_root,
            )
            s_look, l_valid = _look(prim_idx)
            with self._amp_context():
                root_out = model.root_model(
                    x_t_root, known_root, obs_root, s_ee_prim[:, prim_idx], t_root,
                    s_ee_look=s_look, look_valid=l_valid,
                )
            root_losses.append(
                self.root_loss_fn(
                    root_out.pred_v.float(),
                    root_out.pred_x1.float(),
                    s_full_root,
                    v_gt_root,
                    obs_root,
                )
            )
            prev_root_pred = root_out.pred_x1.detach()
            if prim_idx == 0:
                # Segment head = the (possibly perturbed / rollout) seed the
                # chain actually started from, exactly as at inference.
                segment_root_parts.append(history_root.detach())
            segment_root_parts.append(prev_root_pred[:, history_len:])
            del root_out
        # Stage-1 segment estimate, detached: the body stage consumes but must
        # not shape it.
        segment_root = torch.cat(segment_root_parts, dim=1)

        # ── pass 2: body over the whole segment ─────────────────────────────
        seg_anchor = bool(getattr(self.cfg, "ee_cond_segment_anchor", False))
        seg_geometry = bool(getattr(self.cfg, "segment_geometry_loss", False))
        if seg_geometry and segment_full is None:
            raise ValueError(
                "segment_geometry_loss=True requires 'segment_full' in the batch "
                "(G1DeltaFeatPrimitiveDataset emits it)."
            )
        ee_anchor = None
        if seg_anchor:
            # Shared ee_cond anchor: GT hand pose at SEGMENT frame 0 (loss-side
            # only; UNPERTURBED). Same mechanics as the single-stage trainer.
            ee_anchor = self.loss_fn.gt_hand_anchor(s_full_prim[:, 0, :1].float())
        # Non-detached per-primitive body predictions for the stitched segment
        # loss. Their root columns are pinned to the (detached) stage-1 segment
        # root, so segment-level FK gradients still reach the body dofs only --
        # the stage contract ("solve the body given the predicted root") holds
        # at segment level exactly as it does per primitive.
        pred_x1_prims: list[torch.Tensor] = []
        obs_mask_prims: list[torch.Tensor] = []
        body_losses: list[MaskedFlowLossOutput] = []
        prev_body_pred: torch.Tensor | None = None
        for prim_idx in range(prim_cfg.num_primitives):
            prim_start = prim_idx * prim_cfg.future_len
            s_full = s_full_prim[:, prim_idx]
            s_ee = s_ee_prim[:, prim_idx]
            if segment_full is not None and prim_start > 0:
                yaw_offset = segment_full[:, :prim_start, 4].sum(dim=1)
            else:
                yaw_offset = torch.zeros(
                    s_full.shape[0], device=s_full.device, dtype=s_full.dtype
                )
            gt_history = s_full[:, :history_len]
            # Rollout history is the FULL body prediction, whose root columns
            # were pinned to the stage-1 chain -- the root history the body
            # re-seeds from is stage 1's own past output.
            history_in = self._select_history(
                gt_history,
                prev_pred=prev_body_pred,
                train=train,
                force_teacher=(force_teacher or prim_idx == 0),
            )
            history_in = self._perturb_history(history_in, train=train)
            known_body, obs_body = build_prefix_condition_tensors(
                history_in, s_full.shape[1]
            )
            if ee_state_dim > 0:
                apply_ee_state_condition(known_body, obs_body, s_full[..., -ee_state_dim:])
            window_root = segment_root[:, prim_start : prim_start + s_full.shape[1]]
            model.pin_window_root(known_body, obs_body, window_root)
            root_look, root_look_valid = model.build_root_preview(segment_root, prim_start)

            x_t_body, v_gt_body, t_body, _ = model.body_model.sample_training_tuple(
                s_full,
                use_logit_normal_t=use_lnt,
                logit_normal_sigma=lnt_sigma,
                per_frame=per_frame,
                obs_mask=obs_body,
            )
            s_look, l_valid = _look(prim_idx)
            with self._amp_context():
                body_out = model.body_model(
                    x_t_body, known_body, obs_body, s_ee, t_body,
                    s_ee_look=s_look, look_valid=l_valid,
                    root_look=root_look, root_look_valid=root_look_valid,
                )
            body_losses.append(
                self.loss_fn(
                    body_out.pred_v.float(),
                    body_out.pred_x1.float(),
                    s_full,
                    v_gt_body,
                    # Mask WITH the pinned root columns: flow/recon must skip
                    # them (they are stage 1's detached output, not this
                    # stage's prediction).
                    obs_body,
                    s_ee=s_ee,
                    yaw_offset=yaw_offset,
                    anchor_active=(prim_idx == 0),
                    ee_anchor=ee_anchor,
                    include_geometry=not seg_geometry,
                )
            )
            if seg_geometry:
                pred_x1_prims.append(body_out.pred_x1.float())
                obs_mask_prims.append(obs_body)
            prev_body_pred = body_out.pred_x1.detach()
            del body_out

        body_mean = mean_loss_outputs(body_losses)
        if seg_geometry:
            # Stitch the body primitives back into the full segment (drop each
            # later primitive's history overlap). Root columns come out as the
            # single consistent stage-1 segment root (every window pinned a
            # slice of the same tensor), so the stitched geometry/FK loss is
            # the only place a body gradient crosses a primitive seam.
            pred_seg = torch.cat(
                [pred_x1_prims[0]] + [p[:, history_len:] for p in pred_x1_prims[1:]],
                dim=1,
            )
            obs_seg = torch.cat(
                [obs_mask_prims[0]] + [m[:, history_len:] for m in obs_mask_prims[1:]],
                dim=1,
            )
            geom = self.loss_fn.segment_geometry(
                pred_seg,
                segment_full,
                obs_seg,
                s_ee=batch.get("segment_ee"),
                ee_anchor=ee_anchor,
            )
            body_mean = merge_segment_geometry(body_mean, geom)

        return TwoStageLossValues(
            body_mean,
            mean_root_loss_outputs(root_losses),
            stage_weight=float(self.cfg.root_stage.stage_weight),
        )

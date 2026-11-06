"""Synthetic-only image, articulation and metric reconstruction objectives."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .geometry import decode_pose


def heatmap_distribution(
    target_uv: torch.Tensor, height: int, width: int, sigma: float = 1.35
) -> torch.Tensor:
    x = torch.arange(width, device=target_uv.device, dtype=target_uv.dtype) + 0.5
    y = torch.arange(height, device=target_uv.device, dtype=target_uv.dtype) + 0.5
    (yy, xx) = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)
    center = target_uv * target_uv.new_tensor((width, height))
    squared = (grid[None, None] - center[:, :, None, None]).square().sum(-1)
    value = torch.exp(-squared / (2 * sigma * sigma))
    return value / value.sum((-2, -1), keepdim=True).clamp_min(1e-08)


def masked_mean(value, valid):
    return torch.where(valid.bool(), value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(
        1.0
    )


def training_objective(output, target, fk, weight, heatmap_weight):
    supervised_joints = fk.spec.observable_joints
    valid = target["valid"].float()
    true = target["q"].float()
    (lower, upper) = fk.joint_limits.to(output["q"]).unbind(-1)
    q_robust = F.smooth_l1_loss(
        output["q"][:, :supervised_joints], true[:, :supervised_joints], beta=0.04, reduction="none"
    )
    normalized_true = (true - lower) / (upper - lower)
    q_loss = q_robust.mean() + 0.75 * F.smooth_l1_loss(
        output["q_normalized"][:, :supervised_joints],
        normalized_true[:, :supervised_joints],
        beta=0.015,
    )
    uncertainty = F.smooth_l1_loss(
        output["q_log_variance"][:, :supervised_joints],
        q_robust.detach().clamp_min(0.0001).log(),
        beta=0.2,
    )
    uv_true = target["uv"].float()
    safe_uv = torch.where(valid[..., None].bool(), uv_true, torch.zeros_like(uv_true))
    coordinate = F.smooth_l1_loss(
        output["uv_normalized"].float(), safe_uv, beta=0.01, reduction="none"
    ).sum(-1)
    coordinate_loss = masked_mean(coordinate, valid)
    logits = output["heatmap_logits"].float()
    distribution = heatmap_distribution(safe_uv, logits.shape[-2], logits.shape[-1])
    heatmap = -(distribution * F.log_softmax(logits.flatten(2), -1).reshape_as(logits)).sum(
        (-2, -1)
    )
    heatmap_loss = masked_mean(heatmap, valid)
    visibility = F.binary_cross_entropy_with_logits(
        output["visibility_logits"].float(), target.get("visible", valid).float()
    )
    source = fk.semantic_points(output["q"])
    with torch.no_grad():
        true_source = fk.semantic_points(true)
    graph_loss = F.smooth_l1_loss(
        source[:, :, None] - source[:, None, :],
        true_source[:, :, None] - true_source[:, None, :],
        beta=0.01,
    )
    decoded = decode_pose(output, fk)
    target_camera = target["camera_points"].float()
    safe_camera = torch.where(
        valid[..., None].bool(), target_camera, torch.zeros_like(target_camera)
    )
    depth_loss = masked_mean(
        F.smooth_l1_loss(decoded["depth"], safe_camera[..., 2], beta=0.02, reduction="none"), valid
    )
    graph_distance = (decoded["camera_points_nonrigid"] - safe_camera).norm(dim=-1)
    camera_graph_loss = masked_mean(graph_distance, valid)
    pose_loss = (decoded["points"] - target_camera).norm(dim=-1).mean()
    homogeneous = torch.einsum("bij,bnj->bni", target["camera"].float(), decoded["points"])
    projected = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(0.05)
    projected = projected / float(target["image"].shape[-1])
    reprojection_loss = masked_mean(
        F.smooth_l1_loss(projected, safe_uv, beta=0.01, reduction="none").sum(-1), valid
    )
    local_regularizer = (output["local_translation_delta"].float() / 0.025).square().mean()
    articulation = (
        8 * q_loss
        + 0.05 * uncertainty
        + 2 * coordinate_loss
        + heatmap_weight * heatmap_loss
        + 0.1 * visibility
        + 8 * graph_loss
    )
    geometry = (
        5 * depth_loss
        + 8 * camera_graph_loss
        + 10 * pose_loss
        + 2 * reprojection_loss
        + 0.001 * local_regularizer
    )
    loss = articulation + weight * geometry
    size = float(target["image"].shape[-1])
    metrics = dict(
        loss=loss.detach(),
        q_loss=q_loss.detach(),
        q_l1_deg=torch.rad2deg(
            (output["q"][:, :supervised_joints] - true[:, :supervised_joints]).abs()
        )
        .mean()
        .detach(),
        uv_l2_px=(
            size * masked_mean((output["uv_normalized"].float() - safe_uv).norm(dim=-1), valid)
        ).detach(),
        heatmap_loss=heatmap_loss.detach(),
        graph_loss=graph_loss.detach(),
        depth_l1_mm=(
            1000 * masked_mean((decoded["depth"] - safe_camera[..., 2]).abs(), valid)
        ).detach(),
        pose_add_mm=(1000 * pose_loss).detach(),
        unweighted_geometry_loss=geometry.detach(),
        articulation_loss=articulation.detach(),
        geometry_weight=loss.detach().new_tensor(weight),
        local_translation_norm_mm=(
            1000 * output["local_translation_delta"].float().norm(dim=-1).mean()
        ).detach(),
        reprojection_l2_px=(size * masked_mean((projected - safe_uv).norm(dim=-1), valid)).detach(),
        visibility_loss=visibility.detach(),
    )
    return (loss, metrics)


def slice_batch(value, active, batch):
    if torch.is_tensor(value):
        return value[active] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {key: slice_batch(item, active, batch) for (key, item) in value.items()}
    if isinstance(value, list):
        return [slice_batch(item, active, batch) for item in value]
    if isinstance(value, tuple):
        return tuple((slice_batch(item, active, batch) for item in value))
    return value


def graph_zero(value):
    if torch.is_tensor(value):
        return value.float().sum() * 0 if value.requires_grad else None
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return None
    values = [graph_zero(item) for item in items]
    values = [item for item in values if item is not None]
    return sum(values) if values else None


METRICS = (
    "loss",
    "q_loss",
    "q_l1_deg",
    "uv_l2_px",
    "heatmap_loss",
    "graph_loss",
    "depth_l1_mm",
    "pose_add_mm",
    "unweighted_geometry_loss",
    "articulation_loss",
    "geometry_weight",
    "local_translation_norm_mm",
    "reprojection_l2_px",
    "visibility_loss",
)


def loss_and_metrics(output, target, fk, weight, heatmap_weight=2.0):
    active = target["valid"].bool().any(-1)
    batch = len(active)
    if batch < 1:
        raise ValueError("Empty optimizer batch")
    fraction = active.float().mean()
    if active.any():
        selected_output = output if active.all() else slice_batch(output, active, batch)
        selected_target = target if active.all() else slice_batch(target, active, batch)
        (loss, metrics) = training_objective(
            selected_output,
            selected_target,
            fk,
            weight,
            heatmap_weight,
        )
        loss = loss * fraction
        metrics = {
            key: value if key == "geometry_weight" else value * fraction
            for (key, value) in metrics.items()
        }
        metrics["loss"] = loss.detach()
    else:
        loss = graph_zero(output)
        if loss is None:
            raise ValueError("No differentiable model output")
        metrics = {key: loss.detach().new_zeros(()) for key in METRICS}
        metrics["geometry_weight"] = loss.detach().new_tensor(weight)
    if set(metrics) != set(METRICS):
        raise RuntimeError("Different loss metric keys would break DDP reduction")
    metrics.update(
        training_supervision_fraction=fraction.detach(),
        empty_training_crop_fraction=(1 - fraction).detach(),
    )
    return (loss, metrics)

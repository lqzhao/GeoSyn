"""Evaluate a frozen GeoSyn checkpoint on complete DREAM subsets."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .checkpoints import file_sha256, load_model
from .config import add_data_arguments, apply_overrides, load_config
from .data import InferenceDataset, scoring_targets
from .geometry import ProjectionDepthPrior, decode_pose
from .kinematics import RobotKinematics
from .runtime import atomic_save, barrier, save_json, setup


def summarize_add(distances):
    values = np.asarray(distances, np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("ADD requires finite predictions for every benchmark frame")
    # Exactly the original 10-micrometer trapezoidal integration grid.
    thresholds = np.arange(0.0, 0.1, 0.00001)
    cdf = np.searchsorted(np.sort(values), thresholds, side="right") / len(values)
    return dict(
        samples=len(values),
        mean_add_mm=float(values.mean() * 1000),
        median_add_mm=float(np.median(values) * 1000),
        standard_add_auc_percent=float(np.trapz(cdf, dx=0.00001) / 0.1 * 100),
        add_10cm_percent=float((values <= 0.1).mean() * 100),
    )


def score_frozen(directory, dataset, world, diagnostic=False):
    """Pose annotations are first opened here, after all rank files exist."""
    rows = []
    digests = {}
    for rank in range(world):
        path = directory / f"predictions_rank{rank:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["ground_truth_accessed"] is not False:
            raise ValueError("Scoring requires predictions saved before label access")
        rows.extend(payload["predictions"])
        digests[path.name] = file_sha256(path)
    rows.sort(key=lambda row: row["index"])
    expected = len(rows) if diagnostic else len(dataset)
    if expected < 1 or [row["index"] for row in rows] != list(range(expected)):
        raise ValueError("Missing, duplicated or reordered benchmark predictions")
    distances, errors = [], []
    robot = dataset.config["robot"]
    observable = RobotKinematics(robot).spec.observable_joints
    for row in rows:
        image = dataset.paths[row["index"]]
        if row["image"] != image.name:
            raise ValueError("Frozen prediction manifest and scoring images differ")
        q, _, xyz = scoring_targets(image, robot)
        distances.append(float((row["points"] - torch.from_numpy(xyz)).norm(dim=-1).mean()))
        errors.append(
            torch.rad2deg((row["q"][:observable] - torch.from_numpy(q[:observable])).abs())
        )
    error = torch.stack(errors)
    per_joint = error.mean(0).tolist()
    columns = (
        error.reshape(len(error), 6, 2).mean((0, 2)).tolist() if robot == "baxter" else per_joint
    )
    return dict(
        subset=dataset.split,
        diagnostic_incomplete=diagnostic,
        official_rigid_add=summarize_add(distances),
        joint_mae_deg=float(error.mean()),
        per_observable_joint_mae_deg=per_joint,
        J1_J6_mae_deg=columns,
        observable_joints=observable,
        excluded_frames=0,
        predictions_frozen_before_labels=True,
        frozen_prediction_sha256=digests,
    )


@torch.inference_mode()
def evaluate_subset(
    model, config, split, out, rank, world, device, batch_size=32, workers=4, limit=None
):
    dataset = InferenceDataset(config, split)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    indices = list(range(count))[rank::world]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    fk, prior = RobotKinematics(config["robot"], device), ProjectionDepthPrior(
        config["robot"], device
    )
    directory = Path(out) / split
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    predictions = []
    started = time.time()
    for batch_number, batch in enumerate(loader):
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in ("image", "camera", "robot_mask")
        }
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        with autocast:
            output = model(inputs["image"], inputs["camera"], inputs["robot_mask"], fk, prior)
            # Preserve the arithmetic scope used by each original evaluation.
            if config["robot"] != "panda":
                decoded = decode_pose(output, fk)
        if config["robot"] == "panda":
            decoded = decode_pose(output, fk)
        tensors = dict(
            points=decoded["points"].float().cpu(),
            q=output["q"].float().cpu(),
            rotation=decoded["rotation"].float().cpu(),
            translation=decoded["translation"].float().cpu(),
        )
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise FloatingPointError(
                "Nonfinite predictions; no frames can be dropped from evaluation"
            )
        for local, index in enumerate(batch["index"].tolist()):
            predictions.append(
                dict(
                    index=index,
                    image=dataset.paths[index].name,
                    **{key: value[local].clone() for key, value in tensors.items()},
                )
            )
        if batch_number % 25 == 0 and rank == 0:
            print(
                f"{split}: rank 0 completed {len(predictions)}/{len(indices)} frames; "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
    atomic_save(
        dict(
            robot=config["robot"],
            subset=split,
            ground_truth_accessed=False,
            diagnostic_incomplete=limit is not None,
            predictions=predictions,
        ),
        directory / f"predictions_rank{rank:03d}.pt",
    )
    barrier()
    result = score_frozen(directory, dataset, world, limit is not None) if rank == 0 else None
    if rank == 0:
        save_json(result, directory / "metrics.json")
        print(
            f'{split}: ADD-AUC={result["official_rigid_add"]["standard_add_auc_percent"]:.4f}%, '
            f'joint MAE={result["joint_mae_deg"]:.4f} deg',
            flush=True,
        )
    barrier()
    return result


def aggregate(results):
    summary = {"subsets": results}
    groups = {
        "all": list(results),
        "synthetic": [key for key in results if key.endswith(("_dr", "_photo"))],
        "real": [key for key in results if not key.endswith(("_dr", "_photo"))],
    }
    for group, keys in groups.items():
        if keys:
            summary[f"{group}_average_add_auc_percent"] = float(
                np.mean(
                    [results[key]["official_rigid_add"]["standard_add_auc_percent"] for key in keys]
                )
            )
    synthetic = groups["synthetic"]
    if synthetic:
        summary["synthetic_average_joint_mae_deg"] = float(
            np.mean([results[key]["joint_mae_deg"] for key in synthetic])
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--subsets", nargs="+", help="Default: every test subset in the configuration"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--limit", type=int, help="Functional check only; marks output as incomplete"
    )
    args = parser.parse_args()
    if min(args.batch_size, args.limit or 1) < 1 or args.workers < 0:
        parser.error("Batch size/limit must be positive; workers must be nonnegative")
    config = apply_overrides(load_config(args.config), args)
    subsets = args.subsets or list(config["splits"])
    if len(set(subsets)) != len(subsets) or not set(subsets) <= set(config["splits"]):
        parser.error("Subsets must be distinct names from the selected robot configuration")
    rank, world, device = setup(requested_device=args.device)
    checkpoint = args.checkpoint or Path(config["checkpoint"])
    model, metadata = load_model(checkpoint, config["robot"], device)
    results = {
        split: evaluate_subset(
            model,
            config,
            split,
            args.out,
            rank,
            world,
            device,
            args.batch_size,
            args.workers,
            args.limit,
        )
        for split in subsets
    }
    if rank == 0:
        summary = aggregate(results)
        summary.update(
            robot=config["robot"],
            checkpoint_sha256=file_sha256(checkpoint),
            checkpoint_epoch=metadata.get("epoch"),
            diagnostic_incomplete=args.limit is not None,
            precision="bfloat16" if device.type == "cuda" else "float32",
            world_size=world,
            batch_size_per_gpu=args.batch_size,
        )
        save_json(summary, args.out / "summary.json")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

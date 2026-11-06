"""Train GeoSyn on the official synthetic DR TRAIN split."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoints import initialize_from_panda
from .config import add_data_arguments, apply_overrides, load_config
from .data import SyntheticTrainingDataset
from .geometry import ProjectionDepthPrior
from .kinematics import RobotKinematics
from .losses import loss_and_metrics
from .model import GeoSyn
from .runtime import atomic_save, barrier, save_json, setup

GEOMETRY_PREFIXES = (
    "joint_ray_projection.",
    "pose_graph.",
    "node_log_depth.",
    "global_log_depth.",
    "root_depth_occupancy.",
    "node_depth_occupancy.",
    "identifiable_root_occupancy.",
    "identifiable_node_occupancy.",
)


def parameter_group(name):
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith(("sample_gate.", "local_translation_head.")):
        return "translation"
    if name.startswith(GEOMETRY_PREFIXES):
        return "geometry"
    return "heads"


def lr_multiplier(step, total_steps, warmup_fraction=0.005, minimum=0.05):
    warmup = max(1, int(warmup_fraction * total_steps))
    if step < warmup:
        return max(minimum, (step + 1) / warmup)
    fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))


def geometry_weight(step, warmup, ramp):
    if step < warmup:
        return 0.0
    return min(1.0, (step - warmup + 1) / max(1, ramp))


def rng_state(device):
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def collect_rng(device, world):
    state = rng_state(device)
    if world == 1:
        return [state]
    states = [None] * world
    dist.all_gather_object(states, state)
    return states


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--batch-size", type=int, help="Per GPU; the paper uses 64 on each of 2 GPUs"
    )
    parser.add_argument("--epochs", type=int, help="Override the planned training/scheduler length")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--max-steps", type=int, help="Diagnostic optimizer-step limit; not a complete training run"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save weights in a diagnostic run"
    )
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args)
    training = config["train"]
    epochs = args.epochs or training["epochs"]
    batch_size = args.batch_size or training["batch_size"]
    if min(epochs, batch_size, args.max_steps or 1) < 1 or args.workers < 0:
        parser.error(
            "Epochs, batch size and max steps must be positive; workers must be nonnegative"
        )
    if args.no_save and args.max_steps is None:
        parser.error("--no-save is restricted to diagnostic --max-steps runs")
    rank, world, device = setup(training["seed"], args.device)
    dataset = SyntheticTrainingDataset(config)
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=training["seed"], drop_last=False
    )
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    fresh_panda = config["robot"] == "panda" and args.resume is None
    if fresh_panda and not Path(config["dino_weight"]).is_file():
        raise FileNotFoundError(
            "Download the public DINOv2 ViT-L/14 initialization to dino_weight in the YAML"
        )
    raw = GeoSyn(
        robot=config["robot"],
        pretrained_dino=fresh_panda,
        dino_weight=config["dino_weight"],
        input_size=config["input_size"],
        trainable_last_blocks=config["trainable_last_blocks"],
    )
    initialization = None
    if config["robot"] != "panda" and args.resume is None:
        initialization = initialize_from_panda(raw, config["initialize_from"])
    raw.to(device)
    model = (
        DistributedDataParallel(
            raw,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        if world > 1
        else raw
    )
    groups = {name: [] for name in ("backbone", "heads", "geometry", "translation")}
    for name, parameter in raw.named_parameters():
        if parameter.requires_grad:
            groups[parameter_group(name)].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            dict(name=name, params=values, lr=training["learning_rates"][name])
            for name, values in groups.items()
            if values
        ],
        weight_decay=training["weight_decay"],
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step, epochs * len(loader), training["warmup_fraction"], training["min_lr_ratio"]
        ),
    )
    fk, prior = RobotKinematics(config["robot"], device), ProjectionDepthPrior(
        config["robot"], device
    )
    start_epoch, global_step = 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            saved.get("diagnostic_incomplete", False)
            or saved["robot"] != config["robot"]
            or saved["world_size"] != world
            or saved["steps_per_epoch"] != len(loader)
            or saved["schedule_epochs"] != epochs
        ):
            raise ValueError(
                "Resume requires the same robot, GPU count, batch size and training schedule"
            )
        raw.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, global_step = saved["epoch"], saved["global_step"]
        restore_rng(saved["rng_by_rank"][rank], device)
        del saved
    if start_epoch >= epochs:
        raise ValueError("The resumed checkpoint has already completed the requested epochs")
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        save_json(
            dict(
                robot=config["robot"],
                unique_synthetic_train_images=len(dataset),
                batch_size_per_gpu=batch_size,
                world_size=world,
                epochs=epochs,
                steps_per_epoch=len(loader),
                initialization=initialization,
                benchmark_images_in_optimizer=False,
                benchmark_evaluation_during_training=False,
                snapshot_epoch=training["reported_epoch"],
                diagnostic_incomplete=args.max_steps is not None,
            ),
            args.out / "training_protocol.json",
        )
    barrier()
    for epoch in range(start_epoch + 1, epochs + 1):
        sampler.set_epoch(epoch)
        generator.manual_seed(training["seed"] + epoch * 100003 + rank)
        model.train()
        sums, samples = {}, 0
        began = time.time()
        stopped_early = False
        for step, batch in enumerate(loader):
            target = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            optimizer.zero_grad(set_to_none=True)
            weight = geometry_weight(
                global_step, training["geometry_warmup_steps"], training["geometry_ramp_steps"]
            )
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(
                    target["image"], target["camera"], target["robot_mask"], fk, prior
                )
                loss, metrics = loss_and_metrics(prediction, target, fk, weight, 2.0)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite)
            if int(finite) != world:
                raise FloatingPointError(f"Nonfinite training loss: epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            global_step += 1
            size = len(target["image"])
            samples += size
            for name, value in metrics.items():
                sums[name] = sums.get(name, value.new_zeros(())) + value.detach() * size
            if rank == 0 and (step % 50 == 0 or step + 1 == len(loader)):
                print(
                    f"epoch {epoch}/{epochs}, step {step + 1}/{len(loader)}, "
                    f'loss={float(sums["loss"] / samples):.5f}, q={float(sums["q_l1_deg"] / samples):.3f} deg, '
                    f"geometry={weight:.3f}, elapsed={time.time() - began:.1f}s",
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped_early = True
                break
        count = torch.tensor(float(samples), device=device)
        if world > 1:
            dist.all_reduce(count)
            for value in sums.values():
                dist.all_reduce(value)
        means = {name: float(value / count) for name, value in sums.items()}
        states = collect_rng(device, world)
        if rank == 0:
            with (args.out / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(epoch=epoch, global_step=global_step, **means)) + "\n")
            if not args.no_save:
                payload = dict(
                    format_version=1,
                    robot=config["robot"],
                    epoch=epoch,
                    model_config=dict(
                        input_size=config["input_size"],
                        trainable_last_blocks=config["trainable_last_blocks"],
                    ),
                    state_dict=raw.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng_by_rank=states,
                    global_step=global_step,
                    steps_per_epoch=len(loader),
                    world_size=world,
                    schedule_epochs=epochs,
                    diagnostic_incomplete=stopped_early,
                    training_metrics=means,
                )
                atomic_save(payload, args.out / ("diagnostic.pt" if stopped_early else "last.pt"))
                if epoch == training["reported_epoch"] and not stopped_early:
                    snapshot = {
                        key: value
                        for key, value in payload.items()
                        if key not in ("optimizer", "scheduler", "rng_by_rank")
                    }
                    atomic_save(snapshot, args.out / f"epoch_{epoch:03d}.pt")
        barrier()
        if stopped_early:
            break
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

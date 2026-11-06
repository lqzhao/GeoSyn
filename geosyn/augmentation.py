"""Camera-consistent crop augmentation for synthetic training observations."""
import cv2
import numpy as np
import torch


def warp_observation(observation, transform, *, synthetic_TRAIN):
    if synthetic_TRAIN is not True:
        raise ValueError("Metric crop augmentation permits SYN TRAIN only")
    if set(observation).intersection({"q", "uv", "valid", "visible", "camera_points"}):
        raise ValueError("Transform input observations before reading TRAIN targets")
    image = observation["image"]
    if (
        image.device.type != "cpu"
        or image.ndim != 3
        or image.shape[0] != 3
        or (image.shape[-1] != image.shape[-2])
    ):
        raise ValueError("Require square normalized CPU RGB observation")
    size = int(image.shape[-1])
    matrix = np.asarray(transform, dtype=np.float32)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or (not np.array_equal(matrix[2], np.array([0.0, 0.0, 1.0], np.float32)))
        or (matrix[0, 1] != 0)
        or (matrix[1, 0] != 0)
        or (matrix[0, 0] <= 0)
        or (matrix[1, 1] != matrix[0, 0])
    ):
        raise ValueError("Require positive isotropic planar crop transform")
    output = dict(observation)
    rgb = image.permute(1, 2, 0).numpy()
    warped = cv2.warpAffine(
        rgb, matrix[:2], (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    output["image"] = torch.from_numpy(warped.copy()).permute(2, 0, 1).contiguous()
    mask = observation["robot_mask"]
    if mask.device.type != "cpu" or mask.ndim != 2 or mask.shape[-1] != mask.shape[-2]:
        raise ValueError("Require square frozen CPU robot mask")
    mask_size = int(mask.shape[-1])
    down = np.diag([mask_size / size, mask_size / size, 1.0]).astype(np.float32)
    mask_transform = down @ matrix @ np.linalg.inv(down)
    warped_mask = cv2.warpAffine(
        mask.numpy(),
        mask_transform[:2],
        (mask_size, mask_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    output["robot_mask"] = torch.from_numpy(warped_mask.copy()).to(mask)
    tensor = torch.from_numpy(matrix)
    output["camera"] = tensor @ observation["camera"]
    output["affine"] = tensor @ observation["affine"]
    output["synthetic_metric_crop_transform"] = tensor
    return output


def random_metric_crop(observation, rng, *, synthetic_TRAIN, probability=0.75):
    if synthetic_TRAIN is not True or not 0 <= probability <= 1:
        raise ValueError("Require SYN TRAIN and a valid augmentation probability")
    if set(observation).intersection({"q", "uv", "valid", "visible", "camera_points"}):
        raise ValueError("No targets permitted in crop decisions")
    size = int(observation["image"].shape[-1])
    scale = (
        float(np.exp(rng.uniform(np.log(0.75), np.log(1.5)))) if rng.random() < probability else 1.0
    )
    shift = rng.uniform(-0.04, 0.04, size=2) * size if scale != 1.0 else np.zeros(2)
    center = size * 0.5
    matrix = np.array(
        [
            [scale, 0.0, (1 - scale) * center + shift[0]],
            [0.0, scale, (1 - scale) * center + shift[1]],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    return warp_observation(observation, matrix, synthetic_TRAIN=True)

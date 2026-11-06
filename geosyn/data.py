"""DREAM data loading with separate RGB observations and scoring labels."""
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import random_metric_crop
from .checkpoints import file_sha256
from .kinematics import robot_spec

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def image_paths(config, split):
    definition = config["train"] if split == "train" else config["splits"][split]
    root = Path(config["data_root"]) / definition["folder"]
    paths = sorted(root.glob("*.rgb.jpg"))
    expected = definition["count"]
    if len(paths) != expected:
        raise ValueError(
            f"Incomplete or duplicated {split}: {len(paths)} images; expected {expected} in {root}"
        )
    if split == "panda_kinect":
        # Preserve the complete label-free enumeration used by the paper.
        order = (
            (Path(__file__).resolve().parent / "assets/panda_kinect_order.txt")
            .read_text()
            .splitlines()
        )
        if len(order) != len(set(order)) or set(order) != {path.name for path in paths}:
            raise ValueError("Kinect frame names do not match the complete benchmark input order")
        paths = [root / name for name in order]
    return paths


def camera_matrix(folder):
    settings = json.loads((Path(folder) / "_camera_settings.json").read_text())
    values = settings["camera_settings"][0]["intrinsic_settings"]
    matrix = np.asarray(
        [[values["fx"], 0.0, values["cx"]], [0.0, values["fy"], values["cy"]], [0.0, 0.0, 1.0]],
        np.float32,
    )
    if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("Invalid camera calibration")
    return matrix


def annotation_path(image):
    return image.with_name(image.name.removesuffix(".rgb.jpg") + ".json")


def scoring_targets(image, robot):
    """Called by synthetic training or by scoring after predictions are saved."""
    annotation = json.loads(annotation_path(image).read_text())
    spec = robot_spec(robot)
    joints = {
        item["name"].split("/")[-1].replace("iiwa7_", "iiwa_"): float(item["position"])
        for item in annotation["sim_state"]["joints"]
    }
    points = {item["name"]: item for item in annotation["objects"][0]["keypoints"]}
    q = np.asarray([joints[name] for name in spec.joint_names], np.float32)
    uv = np.asarray(
        [points[name]["projected_location"] for name in spec.annotation_links], np.float32
    )
    xyz = np.asarray([points[name]["location"] for name in spec.annotation_links], np.float32)
    if robot != "panda" or "synth" in image.parent.name:
        xyz /= 100.0
    if not np.isfinite(q).all() or not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite scoring/training target: {image}")
    return q, uv, xyz


def crop_affine(box, size):
    x0, y0, x1, y1 = map(float, box)
    width, height = x1 - x0, y1 - y0
    if not np.isfinite(box).all() or min(width, height) < 2:
        raise ValueError(f"Invalid crop box: {box}")
    scale = float(size) / max(width, height)
    resized_w, resized_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    sx, sy = resized_w / width, resized_h / height
    px, py = (size - resized_w) // 2, (size - resized_h) // 2
    return np.asarray(
        [[sx, 0.0, px - sx * x0], [0.0, sy, py - sy * y0], [0.0, 0.0, 1.0]], np.float32
    )


def padded_box(box, image_hw, detected=True, padding=1.35):
    height, width = image_hw
    if not detected:
        return 0.0, 0.0, float(width), float(height)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * padding / 2
    return (
        max(0.0, cx - half),
        max(0.0, cy - half),
        min(float(width), cx + half),
        min(float(height), cy + half),
    )


def cache_path(image, root):
    return Path(root) / image.parent.name / (image.name.removesuffix(".rgb.jpg") + ".npz")


def read_observation(image, root):
    path = cache_path(image, root)
    if not path.is_file():
        raise FileNotFoundError(f"Prepare the frozen RGB observations first: {path}")
    with np.load(path, allow_pickle=False) as payload:
        record = {key: payload[key].copy() for key in payload.files}
    if int(record["schema_version"]) != 1 or str(record["image_sha256"]) != file_sha256(image):
        raise ValueError(f"Observation does not match the RGB bytes: {image}")
    if not bool(record["rgb_only"]):
        raise ValueError("Inference observations must be generated from RGB only")
    return record


def render_observation(image, camera, box, size, mask=None):
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image)
    height, width = bgr.shape[:2]
    affine = crop_affine(box, size)
    rgb = cv2.warpAffine(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        affine[:2],
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if mask is None:
        mask = np.zeros((height, width), np.uint8)
        # Kuka/Baxter use the padded detector-box rectangle as their observation.
        x0, y0, x1, y1 = box
        mask[
            max(0, int(y0)) : min(height, int(np.ceil(y1))),
            max(0, int(x0)) : min(width, int(np.ceil(x1))),
        ] = 1
    if mask.shape != (height, width):
        raise ValueError("Segmentation dimensions differ from the original image")
    cropped_mask = cv2.warpAffine(
        mask,
        affine[:2],
        (size, size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    small_mask = cv2.resize(
        cropped_mask.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST
    )
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return (
        dict(
            image=(tensor - IMAGENET_MEAN) / IMAGENET_STD,
            camera=torch.from_numpy(affine @ np.asarray(camera, dtype=np.float32)),
            robot_mask=torch.from_numpy(small_mask.copy()).float(),
            affine=torch.from_numpy(affine),
        ),
        (height, width),
    )


class InferenceDataset(Dataset):
    """Reads RGB, calibration and frozen detector/mask outputs; never pose labels."""

    def __init__(self, config, split):
        self.config, self.split = config, split
        self.paths = image_paths(config, split)
        self.camera = camera_matrix(self.paths[0].parent)
        self.size = config.get("input_size", 448)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.paths[index]
        record = read_observation(image, self.config["observations_root"])
        box = padded_box(record["box_xyxy"], record["image_hw"], bool(record["detected"]))
        mask = record.get("robot_mask")
        if self.config["robot"] == "panda" and mask is None:
            raise ValueError("Panda requires a RobotSeg mask for every frame")
        sample, hw = render_observation(image, self.camera, box, self.size, mask)
        if tuple(record["image_hw"]) != hw:
            raise ValueError(f"Observation/RGB dimensions differ: {image}")
        sample["index"] = index
        return sample


class SyntheticTrainingDataset(Dataset):
    """Only the named synthetic TRAIN directory may contribute to optimization."""

    def __init__(self, config):
        self.config = config
        folder = config["train"]["folder"]
        if "synth_train_dr" not in Path(folder).name:
            raise ValueError("Optimization requires the official synthetic DR TRAIN split")
        self.paths = image_paths(config, "train")
        self.camera = camera_matrix(self.paths[0].parent)
        self.size = config.get("input_size", 448)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.paths[index]
        robot = self.config["robot"]
        if robot == "panda":
            record = read_observation(image, self.config["observations_root"])
            box = padded_box(record["box_xyxy"], record["image_hw"], bool(record["detected"]))
            sample, hw = render_observation(
                image, self.camera, box, self.size, record["robot_mask"]
            )
        else:
            # Synthetic annotation boxes are allowed for Kuka/Baxter TRAIN only.
            annotation = json.loads(annotation_path(image).read_text())
            bounds = annotation["objects"][0]["bounding_box"]
            box = padded_box((*bounds["min"], *bounds["max"]), (480, 640))
            if not np.isfinite(box).all() or min(box[2] - box[0], box[3] - box[1]) < 2:
                box = (0.0, 0.0, 640.0, 480.0)
            sample, hw = render_observation(image, self.camera, box, self.size)
        sample = random_metric_crop(sample, np.random, synthetic_TRAIN=True, probability=0.75)
        q, uv, xyz = scoring_targets(image, robot)
        height, width = hw
        transformed = (
            np.concatenate((uv, np.ones((len(uv), 1), np.float32)), 1) @ sample["affine"].numpy().T
        )
        valid = (
            np.isfinite(uv).all(1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
            & (transformed[:, 0] >= 0)
            & (transformed[:, 0] < self.size)
            & (transformed[:, 1] >= 0)
            & (transformed[:, 1] < self.size)
        )
        sample.update(
            q=torch.from_numpy(q),
            uv=torch.from_numpy(transformed[:, :2] / self.size),
            valid=torch.from_numpy(valid),
            visible=torch.from_numpy(valid.copy()),
            camera_points=torch.from_numpy(xyz),
        )
        return sample

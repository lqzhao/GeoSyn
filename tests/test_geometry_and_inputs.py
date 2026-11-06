import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from geosyn.checkpoints import file_sha256
from geosyn.data import InferenceDataset, cache_path, scoring_targets
from geosyn.evaluate import summarize_add
from geosyn.geometry import differentiable_weighted_kabsch, weighted_kabsch
from geosyn.kinematics import RobotKinematics
from geosyn.prepare import load_mask


class GeometryAndInputTests(unittest.TestCase):
    def test_rigid_alignment_recovers_known_transform(self):
        torch.manual_seed(3)
        source = torch.randn(2, 17, 3)
        angle = torch.tensor(0.4)
        rotation = torch.tensor(
            [[angle.cos(), -angle.sin(), 0.0], [angle.sin(), angle.cos(), 0.0], [0.0, 0.0, 1.0]]
        )[None].repeat(2, 1, 1)
        translation = torch.tensor([[0.2, -0.3, 1.0], [-0.1, 0.4, 2.0]])
        target = torch.einsum("bij,bnj->bni", rotation, source) + translation[:, None]
        estimated_r, estimated_t = weighted_kabsch(source, target, torch.rand(2, 17) + 0.1)
        torch.testing.assert_close(estimated_r, rotation, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(estimated_t, translation, atol=1e-6, rtol=1e-6)

    def test_alignment_has_finite_gradients_for_collinear_points(self):
        source = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]], requires_grad=True
        )
        target = source.detach() + torch.tensor([0.1, -0.1, 1.0])
        rotation, translation = differentiable_weighted_kabsch(source, target, torch.ones(1, 3))
        (rotation.square().sum() + translation.square().sum()).backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        torch.testing.assert_close(torch.det(rotation), torch.ones(1), atol=1e-6, rtol=1e-6)

    def test_all_robot_chains_produce_expected_keypoint_counts(self):
        for robot, count in [("panda", 7), ("kuka", 8), ("baxter", 17)]:
            fk = RobotKinematics(robot)
            q = fk.joint_limits.mean(-1)[None].requires_grad_(True)
            points = fk.semantic_points(q)
            self.assertEqual(tuple(points.shape), (1, count, 3))
            points.square().sum().backward()
            self.assertTrue(torch.isfinite(q.grad).all())

    def test_auc_matches_original_trapezoidal_definition(self):
        values = np.asarray([0.0, 0.001, 0.035, 0.09999, 0.2], np.float64)
        thresholds = np.arange(0.0, 0.1, 0.00001)
        expected = np.trapz((values[:, None] <= thresholds[None]).mean(0), dx=0.00001) / 0.1 * 100
        self.assertEqual(summarize_add(values)["standard_add_auc_percent"], expected)

    def test_inference_does_not_open_frame_pose_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "kuka_synth_test_dr"
            images.mkdir()
            image = images / "000000.rgb.jpg"
            cv2.imwrite(str(image), np.full((48, 64, 3), 128, np.uint8))
            (images / "000000.json").write_text("THIS IS DELIBERATELY NOT A POSE ANNOTATION")
            (images / "_camera_settings.json").write_text(
                json.dumps(
                    {
                        "camera_settings": [
                            {
                                "intrinsic_settings": {
                                    "fx": 100.0,
                                    "fy": 100.0,
                                    "cx": 32.0,
                                    "cy": 24.0,
                                }
                            }
                        ]
                    }
                )
            )
            observations = root / "observations"
            observation = cache_path(image, observations)
            observation.parent.mkdir(parents=True)
            np.savez(
                observation,
                schema_version=np.int64(1),
                image_sha256=np.asarray(file_sha256(image)),
                rgb_only=np.bool_(True),
                image_hw=np.asarray([48, 64]),
                box_xyxy=np.asarray([0.0, 0.0, 64.0, 48.0]),
                detected=np.bool_(False),
            )
            config = dict(
                robot="kuka",
                data_root=str(root),
                observations_root=str(observations),
                splits={"kuka_dr": dict(folder=images.name, count=1)},
                input_size=448,
            )
            with patch(
                "geosyn.data.scoring_targets", side_effect=AssertionError("Inference read labels")
            ):
                sample = InferenceDataset(config, "kuka_dr")[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 448, 448))
            self.assertEqual(tuple(sample["robot_mask"].shape), (64, 64))

    def test_kuka_annotation_names_and_centimeters_match_urdf_convention(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "000000.rgb.jpg"
            annotation = dict(
                sim_state={
                    "joints": [
                        dict(name=f"robot/iiwa7_joint_{i}", position=0.1 * i) for i in range(1, 8)
                    ]
                },
                objects=[
                    dict(
                        keypoints=[
                            dict(
                                name=f"iiwa7_link_{i}",
                                projected_location=[10.0, 20.0],
                                location=[0.0, 0.0, 100.0 + i],
                            )
                            for i in range(8)
                        ]
                    )
                ],
            )
            (Path(folder) / "000000.json").write_text(json.dumps(annotation))
            q, uv, xyz = scoring_targets(image, "kuka")
            np.testing.assert_allclose(q, np.arange(1, 8) * 0.1, rtol=1e-6)
            self.assertEqual(uv.shape, (8, 2))
            np.testing.assert_allclose(xyz[:, 2], np.arange(100, 108) / 100.0, rtol=1e-6)

    def test_mask_import_cannot_mix_equal_frame_ids_from_different_subsets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            masks = root / "masks"
            masks.mkdir()
            image = root / "panda_synth_test_dr/000000.rgb.jpg"
            np.savez(
                masks / "000000.npz",
                robot_mask=np.ones((4, 4), np.uint8),
                img_path=np.asarray("/data/panda-3cam_kinect360/raw_4966/000000.rgb.jpg"),
            )
            with self.assertRaises(FileNotFoundError):
                load_mask(image, [masks])
            np.savez(
                masks / "000000.npz",
                robot_mask=np.ones((4, 4), np.uint8),
                img_path=np.asarray("/data/synthetic/panda_synth_test_dr/000000.rgb.jpg"),
            )
            self.assertEqual(load_mask(image, [masks]).shape, (4, 4))


if __name__ == "__main__":
    unittest.main()

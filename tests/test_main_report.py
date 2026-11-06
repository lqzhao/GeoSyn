import json
from pathlib import Path
import tempfile
import unittest

from geosyn.report import EXPECTED_SPLITS, combine


class MainReportTests(unittest.TestCase):
    def make_results(self, root):
        synthetic_error = {
            "panda_dr": 1.0,
            "panda_photo": 2.0,
            "kuka_dr": 3.0,
            "kuka_photo": 4.0,
            "baxter_dr": 5.0,
        }
        score = 10.0
        for robot, subsets in EXPECTED_SPLITS.items():
            rows = {}
            for subset, count in subsets.items():
                error = synthetic_error.get(subset, 0.0)
                rows[subset] = dict(
                    diagnostic_incomplete=False,
                    official_rigid_add=dict(samples=count, standard_add_auc_percent=score),
                    J1_J6_mae_deg=[error] * 6,
                    joint_mae_deg=error,
                )
                score += 10.0
            directory = root / robot
            directory.mkdir()
            (directory / "summary.json").write_text(
                json.dumps(
                    dict(diagnostic_incomplete=False, checkpoint_sha256="fixture", subsets=rows)
                )
            )

    def test_subset_means_are_equal_weighted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_results(root)
            report = combine(root)
            self.assertEqual(report["panda_synthetic_average"], 15.0)
            self.assertEqual(report["panda_real_average"], 45.0)
            self.assertEqual(report["panda_all_average"], 35.0)
            self.assertEqual(report["five_subset_joint_average"], 3.0)
            self.assertEqual(report["five_subset_J1_J6_average"], [3.0] * 6)

    def test_incomplete_or_diagnostic_scores_cannot_form_main_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_results(root)
            path = root / "baxter/summary.json"
            data = json.loads(path.read_text())
            data["subsets"]["baxter_dr"]["official_rigid_add"]["samples"] = 8
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                combine(root)
            data["subsets"]["baxter_dr"]["official_rigid_add"]["samples"] = 5982
            data["diagnostic_incomplete"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                combine(root)


if __name__ == "__main__":
    unittest.main()

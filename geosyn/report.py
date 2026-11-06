"""Combine the three robot evaluations into the main-experiment results."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import save_json

EXPECTED_SPLITS = {
    "panda": {
        "panda_dr": 5997,
        "panda_photo": 5997,
        "panda_azure": 6394,
        "panda_kinect": 4966,
        "panda_realsense": 5944,
        "panda_orb": 32315,
    },
    "kuka": {"kuka_dr": 5997, "kuka_photo": 5999},
    "baxter": {"baxter_dr": 5982},
}


def combine(root):
    tables, joints, checkpoints = {}, {}, {}
    for robot, expected in EXPECTED_SPLITS.items():
        summary = json.loads((root / robot / "summary.json").read_text())
        if summary["diagnostic_incomplete"] or set(summary["subsets"]) != set(expected):
            raise ValueError(f"{robot} requires all complete subsets before reporting main results")
        checkpoints[robot] = summary["checkpoint_sha256"]
        for split in expected:
            row = summary["subsets"][split]
            if (
                row["diagnostic_incomplete"]
                or row["official_rigid_add"]["samples"] != expected[split]
            ):
                raise ValueError(f"{split} has not evaluated the complete official subset")
            tables[split] = row["official_rigid_add"]["standard_add_auc_percent"]
            if split.endswith(("_dr", "_photo")):
                joints[split] = dict(J1_J6=row["J1_J6_mae_deg"], average=row["joint_mae_deg"])
    panda_real = ["panda_azure", "panda_kinect", "panda_realsense", "panda_orb"]
    return dict(
        ADD_AUC_percent=tables,
        panda_real_average=float(np.mean([tables[key] for key in panda_real])),
        panda_synthetic_average=float(np.mean([tables["panda_dr"], tables["panda_photo"]])),
        panda_all_average=float(np.mean([tables[key] for key in EXPECTED_SPLITS["panda"]])),
        joint_MAE_degrees=joints,
        five_subset_joint_average=float(np.mean([row["average"] for row in joints.values()])),
        five_subset_J1_J6_average=np.mean(
            [row["J1_J6"] for row in joints.values()], axis=0
        ).tolist(),
        checkpoint_sha256=checkpoints,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory containing panda/, kuka/ and baxter/"
    )
    args = parser.parse_args()
    result = combine(args.root)
    save_json(result, args.root / "main_results.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

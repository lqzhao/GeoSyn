"""Run the GeoSyn rows of the main quantitative experiments on one GPU."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    for robot in ["panda", "kuka", "baxter"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "geosyn.evaluate",
                "--config",
                str(args.config_dir / f"{robot}.yaml"),
                "--out",
                str(args.out / robot),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ],
            check=True,
        )
    subprocess.run([sys.executable, "-m", "geosyn.report", "--root", str(args.out)], check=True)


if __name__ == "__main__":
    main()

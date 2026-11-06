"""Verify the packaged geometry and exported checkpoint SHA-256 checksums."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assets = ROOT / "geosyn/assets"
    manifest = json.loads((assets / "manifest.json").read_text())
    for robot, record in manifest.items():
        for suffix, key in [(".urdf", "urdf_sha256"), ("_depth_samples.npz", "samples_sha256")]:
            path = assets / f"{robot}{suffix}"
            if sha256(path) != record[key]:
                raise ValueError(f"Geometry checksum mismatch: {path.name}")
    checkpoints = ROOT / "checkpoints"
    for record in json.loads((checkpoints / "manifest.json").read_text()).values():
        path = checkpoints / record["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Place the model release asset at {path}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
        print(f"{path.name}: verified")
    print("All geometry and model assets verified.")


if __name__ == "__main__":
    main()

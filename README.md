# GeoSyn

**Learning Geometry-Grounded Articulated Robot State Estimation from Synthetic Data**

A standalone implementation of the GeoSyn main experiments on DREAM Panda,
Kuka iiwa and Baxter. It includes camera-aware DINOv2 prediction,
articulation-guided metric depth, weighted 3D alignment, synthetic training,
and complete-subset evaluation of ADD-AUC and joint-angle MAE.

The package runs independently of the original research workspace. Historical
experiment versions, baseline implementations, ablations and figure-generation
tools are outside the scope of this release.

## 1. Environment

The reference environment is:

| Component | Version / requirement |
|---|---|
| OS | Linux |
| Python | 3.10; tested with 3.10.20 |
| PyTorch / torchvision | 2.4.1 / 0.19.1 |
| PyTorch CUDA runtime | 12.1 |
| GPU | NVIDIA Ampere or newer, with BF16 support |
| timm | 1.0.28 |
| NumPy | 1.26.4 |
| pytorch-kinematics | 0.10.0 |

Other dependencies are pinned in [requirements.txt](requirements.txt).
The training recipe uses **2 GPUs with 48 GB each**, a per-GPU batch of 64,
448 × 448 images, and four trainable DINOv2 blocks. Evaluation defaults to a
per-GPU batch of 32. Reduce `--batch-size` when using less GPU memory.

From this directory:

```bash
conda create -n geosyn python=3.10 -y
conda activate geosyn
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

The CUDA wheel needs a compatible NVIDIA driver. A CUDA compiler is only
needed if building the optional external preprocessing tools.

## 2. Data

Obtain the original [DREAM datasets](https://github.com/NVlabs/DREAM).
Preserve the original RGB images, annotations and camera calibration files:

```text
data/
├── panda/
│   ├── synthetic/
│   │   ├── panda_synth_train_dr/      # 104972 images
│   │   ├── panda_synth_test_dr/       # 5997
│   │   └── panda_synth_test_photo/    # 5997
│   └── real/
│       ├── panda-3cam_azure/          # 6394
│       ├── panda-3cam_kinect360/      # 4966
│       ├── panda-3cam_realsense/      # 5944
│       └── panda-orb/                # 32315
├── kuka/
│   ├── kuka_synth_train_dr/           # 104977
│   ├── kuka_synth_test_dr/            # 5997
│   └── kuka_synth_test_photo/         # 5999
├── baxter/
│   ├── baxter_synth_train_dr/         # 104982
│   └── baxter_synth_test_dr/          # 5982
└── observations/                     # generated RGB-only caches
```

Each image directory must contain `_camera_settings.json`, `*.rgb.jpg`, and
the corresponding per-frame `*.json` annotations. You may use directory
symlinks or edit `data_root` in `configs/{panda,kuka,baxter}.yaml`.
**Paths inside a YAML file are relative to that YAML file.**
The `--data-root` and `--observations-root` CLI options override these paths.

FK-only URDF files and the fixed collision samples needed by the depth prior
are already included in `geosyn/assets/`. Their original licenses and
modifications are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 3. Frozen RGB observations

Both training and evaluation consume cached, image-derived observations.
The fixed detector is GroundingDINO Swin-T OGC, with prompt `robotic arm .`,
threshold 0.25, highest-confidence box selection, padding 1.35 and full-image
fallback if detection fails.

Panda additionally uses RobotSeg's `robot` mask, predicted independently for
each RGB image with threshold zero. Kuka and Baxter use the padded box
rectangle for the projection-depth evidence. Their synthetic TRAIN crops
come from synthetic annotation boxes; their TEST crops use RGB detections.

### Import existing observations

If the original frozen GroundingDINO and RobotSeg caches are available,
convert them without rerunning either external model:

```bash
python -m geosyn.prepare --config configs/panda.yaml \
  --import-boxes /path/to/original_groundingdino_panda \
  --import-masks /path/to/original_robotseg_masks
```

Repeat `--import-masks` to search several mask directories. For Kuka and
Baxter, use their configurations and `--import-boxes` only. Imported detector
caches must carry the original RGB SHA-256 and the RGB-only detector protocol.
The new cache contains no absolute image paths and can move with the dataset.

## 4. Model weights

Place the supplied weights in `checkpoints/`:

| File | Purpose | Epoch |
|---|---|---|
| `panda.pt` | Main Panda evaluation; Baxter initialization | 18 |
| `kuka.pt` | Main Kuka evaluation | 28 |
| `baxter.pt` | Main Baxter evaluation | 30 |
| `panda_kuka_init.pt` | Panda initialization used by the Kuka experiment | 18 |

The locally prepared folder includes these inference-only files. They are
excluded from Git and should be distributed as separate release assets.
Their sizes and SHA-256 checksums are recorded in
[checkpoints/manifest.json](checkpoints/manifest.json).
See [checkpoints/README.md](checkpoints/README.md) for checkpoint details.

```bash
python scripts/check_assets.py
```

Training Panda from scratch also requires the public DINOv2 ViT-L/14 weights:

```bash
curl -L --output checkpoints/dinov2_vitl14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

## 5. Reproduce the main evaluations

After configuring the data and preparing the observations:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_main.py \
  --out runs/main --batch-size 32 --workers 4
```

This evaluates all nine test subsets and writes
`runs/main/main_results.json`, with ADD-AUC, joint-angle errors and equal-subset
averages. Individual evaluation commands are also available:

```bash
python -m geosyn.evaluate --config configs/panda.yaml --out runs/main/panda
python -m geosyn.evaluate --config configs/kuka.yaml --out runs/main/kuka
python -m geosyn.evaluate --config configs/baxter.yaml --out runs/main/baxter
python -m geosyn.report --root runs/main
```

For multi-GPU inference, launch one robot evaluation with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m geosyn.evaluate --config configs/panda.yaml --out runs/main/panda
```

Each rank first saves its predictions. Pose annotations are opened only in
the subsequent scoring phase. Evaluation requires every frame in each
selected subset; failed detections are retained using full-image crops.
There is one model per robot, with one shared Panda checkpoint across all
six Panda subsets.

Reference ADD-AUC values (%), rounded to one decimal:

| Robot | DR | Photo | Azure | Kinect | RealSense | ORB |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 82.0 | 85.2 | 80.7 | 80.4 | 79.7 | 77.2 |
| Kuka | 76.0 | 81.6 | — | — | — | — |
| Baxter | 66.9 | — | — | — | — | — |

## 6. Synthetic training

All robot-task optimization uses synthetic DR TRAIN images. The configurations
contain the original learning rates, seeds, warm-up schedule and 30-epoch
learning-rate schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m geosyn.train --config configs/panda.yaml --out runs/train_panda
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m geosyn.train --config configs/kuka.yaml --out runs/train_kuka
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m geosyn.train --config configs/baxter.yaml --out runs/train_baxter
```

Panda starts from public DINOv2 weights. Kuka and Baxter initialize task and
backbone weights from their specified Panda snapshots, while creating fresh
robot-node embeddings and loading their own joint limits. The Kuka experiment
uses an earlier Panda snapshot (`panda_kuka_init.pt`); it must not be silently
replaced with the main Panda weights. To run a new transfer experiment, change
`initialize_from` explicitly in the relevant YAML.

The trainer writes `last.pt`, `training_metrics.jsonl`, and a weights-only
snapshot at the configured reported epoch (`epoch_018.pt`, `epoch_028.pt` or
`epoch_030.pt`). Resume with `--resume runs/train_panda/last.pt`, retaining the
same batch size, GPU count and schedule. Use `--checkpoint` when evaluating a
newly trained snapshot. Reserve at least 6 GB of output disk space per run.


## License

GeoSyn source code is released under the [MIT license](LICENSE). Included
robot resources and external dependencies retain their original licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

# FastViT MedPackID Object Detection

FastViT ImageNet-pretrained backbone + FPN + Faster R-CNN for the MedPackID object detection dataset.

## DGX setup

Clone this repository on DGX under the RAID project directory, then create the Conda environment:

```bash
cd /home/D13K48009/raid/Clara/fastvit-medpack
bash scripts/setup_conda.sh
```

The environment pins Python 3.10, PyTorch 2.5.1/Torchvision 0.20.1 with CUDA 12.1, and MKL below 2025 to avoid the `iJIT_NotifyEvent` ABI mismatch. Python packages are listed in `requirements-dgx.txt`. Verify the DGX driver supports CUDA 12.1 runtime compatibility. The setup installs framework and Python dependencies; it does not download training data.

## Download and train

```bash
cd /home/D13K48009/raid/Clara/fastvit-medpack
conda activate fastvit_medpack
GPU_ID=0 bash run_dgx.sh
```

The runner verifies the supplied SHA-256 for the Download All archive, extracts its nested image and annotation ZIPs, and starts 100 epochs. By default it downloads directly from Mendeley and saves data to `data/MedPackID`; checkpoints/logs go under `runs/fastvit_medpack_100ep`.

If the archive has already been copied from Google Drive to `~/xxdrive/MedPackID/MedPackID_v1.zip`, reuse it without downloading from Mendeley:

```bash
mkdir -p "$HOME/xxdrive/MedPackID"
MEDPACK_DATA_ROOT="$HOME/xxdrive/MedPackID" RUN_DIR="$HOME/xxdrive/MedPackID/runs/fastvit_medpack_100ep" \
  bash run_dgx.sh
```

The script checks the archive checksum and extracts the outer and nested ZIPs into `MEDPACK_DATA_ROOT` before training.

Optional overrides:

```bash
GPU_ID=0 BATCH_SIZE=8 WORKERS=12 IMAGE_SIZE=320 EPOCHS=100 \
MEDPACK_DATA_ROOT=/raid/path/MedPackID RUN_DIR=/raid/path/fastvit-runs \
bash run_dgx.sh
```

To run in the background and keep a log:

```bash
mkdir -p runs/fastvit_medpack_100ep
nohup bash run_dgx.sh > runs/fastvit_medpack_100ep/train.log 2>&1 < /dev/null &
echo $! > runs/fastvit_medpack_100ep/train.pid
tail -f runs/fastvit_medpack_100ep/train.log
```

Training starts at the configured learning rate with no warmup. It saves `last.pth` every epoch, updates `best.pth` by validation loss, and writes `history.json`. It does not evaluate the test split during training.

The dataset archive is large (about 3.03 GB). Ensure the RAID has enough space for the downloaded archive, extracted files, and checkpoints. The supplied checksum is checked before extraction.

## Audit bounding boxes before training

Run the annotation audit before starting or restarting an experiment:

```bash
python src/train_fastvit_medpack.py --data data/MedPackID --audit-only --out runs/annotation_audit
```

It writes `dataset_audit.json` with image counts, positive/empty image counts, and box counts per class, plus `train_target_0.jpg` through `train_target_3.jpg` for visual inspection. Missing or ambiguous image/annotation pairs, invalid coordinates, unknown classes, and a split with zero boxes stop the run. The default prioritizes the YOLO export; `--annotation-format voc` selects the VOC export.

Loss values use scientific notation. A small validation loss is not a detection accuracy metric; use an independent object detection evaluation for final reporting. Start a new output directory after correcting an annotation issue so earlier checkpoints are preserved.

## Evaluate the trained checkpoint

Activate the existing environment and install the COCO metric dependency (no environment rebuild):

```bash
conda activate fastvit_medpack
python -m pip install 'pycocotools>=2.0.7,<3'
CUDA_VISIBLE_DEVICES=0 python -u src/evaluate_fastvit_medpack.py \
  --data data/MedPackID \
  --checkpoint runs/fastvit_medpack_fixed/best.pth \
  --split valid --out runs/eval_valid_best
```

The evaluator reconstructs the architecture and image size from the checkpoint and loads all model weights strictly. It does not download pretrained weights or modify the checkpoint. Full-precision inference uses the same aspect-preserving resize and normalization as training.

Outputs:

- `metrics.json`: checkpoint epoch and SHA-256, split, annotation audit, overall COCO mAP50-95, mAP50, mAP75 and AR100, per-class AP, and precision/recall/F1.
- `per_class.csv`: metrics for `kemasan` and `pil`.
- `previews/*.jpg`: ground truth on the left in green, predictions on the right in red.
- `predictions.json` and `ground_truth.json`: COCO-format records for reproducing the evaluation. `image_index.json` and `preview_index.json` map results to source images.

AP is computed using the official [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) across IoU thresholds 0.50:0.05:0.95, with at most 100 detections per image and an inference score floor of 0.001. Precision/recall/F1 use confidence >= 0.50 and same-class, one-to-one IoU >= 0.50 matching. The confidence threshold applies to precision/recall and previews; AP uses the full set of predictions above the inference floor. Values are on a 0-1 scale, and undefined metrics for classes absent from a split are JSON null.

Use validation to inspect predictions and choose any thresholds. Once the checkpoint, thresholds, and settings are fixed, run the final test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -u src/evaluate_fastvit_medpack.py \
  --data data/MedPackID \
  --checkpoint runs/fastvit_medpack_fixed/best.pth \
  --split test --out runs/eval_test_best
```

The default precision/recall thresholds can be specified with `--score-threshold` and `--iou-threshold`. Keep these settings fixed for test reporting.

## Archive Experiment 01 to xxdrive

After training and validation evaluation finish:

```bash
conda activate fastvit_medpack
bash scripts/backup_exp01.sh
```

Default destination: `~/xxdrive/MedPackID/experiments/exp01_fastvit_320`. An explicit path or rclone remote can be passed as the first argument. The script prepares `runs/archives/exp01_fastvit_320` with:

- The training run (`best.pth`, `last.pth`, loss history, log, dataset audit and target previews).
- Validation metrics, predictions, ground truth and prediction previews.
- A tracked-code snapshot at the archive commit and environment exports captured at archive time.
- `experiment.json` naming the experiment, best epoch and validation results, plus `SHA256SUMS` for every archived file.

It verifies that the checkpoint matches the SHA-256 recorded by evaluation before preparing the archive. Subsequent runs verify and reuse the same local snapshot. rclone copies with progress and `--immutable`, then checks source and destination contents. When `xxdrive` is a cached mount, this verifies the mounted destination; cloud upload completion is managed separately by the mount.

# FOCI Counter  - User Documentation

## Overview

The purpose of this project is to increase accuracy and speed in automated radiobiological foci counting. This is important for biological dosimetry in case of radiation exposure and in fundamental research into the effects of radiation on cells.
The developed models count red $\gamma$-H2AX foci within blue-stained nuclei in fluorescence microscopy images, corresponding to sites of radiation-induced DNA double-strand breaks.
Accuracy of the model is crucial as it reduces the number of cells needed to analyze per blood sample.



## What This App Does

The FOCI Counter GUI processes microscope images and estimates how many red foci are visible per image.
It supports two independent model approaches:

- Object detection (`YOLO`)
- Density-map regression (`SAUnet`-based model)

You can run one model or both and export results as a `.csv`.

## GUI Usage (How-To)

1. Set **Image Directory** (input images).
2. Set **Output Directory** (results).
3. Choose **Models**:
   - `Both`
   - `YOLO`
   - `Densitymap`
4. Choose **Excel Format**:
   - `german` (decimal `,`, separator `;`)
   - `english` (decimal `.`, separator `,`)
5. Optionally enable **Write result images**. This significantly increases the runtime and storage requirements. We recommend to enable this for a small number of images when trying to understand the model's behavior. 
6. Click **Run**.

For each run, the app creates a timestamped folder under the selected output directory, including:

- `foci_counts.csv`
- `foci_counter_log.log`
- optional result images (if enabled)

## CSV Output Explained (`foci_counts.csv`)

The CSV is written by `src/countcv/focidet/main.py`.  
It contains the following columns:

- `filename`: image file path
- `ground_truth`: reference count from available labels
- `yolo`: predicted count from YOLO
- `density`: predicted count from density-map model
- `ensemble`: combined prediction (when both models are available)
- `yolo_interval_min`: lower YOLO count bound from confidence-threshold interval logic
- `yolo_interval_max`: upper YOLO count bound from confidence-threshold interval logic

If only one model is selected, columns for the other model can be empty depending on run configuration.

## How The Two Approaches Work

### YOLO (Object Detection)

YOLO looks for each individual focus and draws a bounding box around it. The [ultralytics framework](https://docs.ultralytics.com/models/yolov8/) is used for this approach. 
The final count is simply the number of detected boxes.

### Density Map (SAUnet)

Instead of detecting each dot explicitly, the model predicts an "intensity map" where brighter regions indicate stronger focus presence.  
Summing the pixel values in the map gives the total count. See [SAU-Net: A Unified Network for Cell Counting in 2D and 3D Microscopy Images](https://doi.org/10.1109/TCBB.2021.3089608) for more details. 

### Combined Output (`ensemble`)

When both models are run, an additional combined estimate (mean) is produced in the output table.

## Existing Models In This Project

- Object detection model family: `YOLOv8` (base version `yolov8n.pt`)
- Density-map model family: `cnn_base` configuration with version `SAUnet`

In the GUI inference workflow, models are loaded from the checkpoint structure (e.g. `checkpoints/yolo/best.pt` and `checkpoints/cnn_base/model.pt`).

## Data Basis

The prediction models were trained on the following data basis:

- Lymphocytes isolated from whole blood samples.
- Dose levels (Gy): `0`, `0.1`, `0.5`, `1`, `2`, `4`
- For each dose: approximately 30 images from `Galerie 1` and `Galerie 3` were manually labelled with bounding boxes 
- Additionally approximately 200 images from Gallery 5 (only 0.5 Gy, 1 Gy) were manually labelled.
- `Rejected` images were excluded from training.

Current split size:

- Train images: `284`
- Validation images: `72`
- Test images: `90`
- Train + Validation total: `356`

## Preprocessing Pipeline

The training/inference preparation includes:

1. Preprocessing:
   - center-crop to `220x220`
2. Stratified split by object count:
   - fixed 20% test split
   - train/val split from remaining data

## Test Metrics

Comparison of one density-map model and one YOLO model (seed=0):

| Metric | SAUnet (Density Map) | YOLO (`yolov8n.pt`) |
|---|---:|---:|
| `mean_count` | 4.43 | 4.43 |
| `mae` (mean absolute error)| 0.79 | 0.81 |
| `mean_error` | -0.02 | -0.06 |
| `mape` (mean absolute percentage error) | 0.13 | 0.14 |
| `error_variance` | 1.60 | 1.81 |
| `number test images` | 90 | 90 |
| `number train/validation images` | 356 | 356 |

Short interpretation:

- YOLO has slightly higher MAE and MAPE in this evaluation.
- `mean_error` indicates both model tend to undercount slightly

### Prediction Intervals for yolo model
Test data evaluation of interval defined by *yolo_interval_min* and *yolo_interval_max* 
- The interval covers 89% of the test data ground truth predictions. 
- The median interval width is 2.
- 7% of the predictions are below the interval limit.
- 4% of the predictions are above the interval limit

## Contact

For questions or feedback, please contact either 
- Tilman Hartwig at tilman.hartwig@uba.de
- ki-anwendungslabor@uba.de

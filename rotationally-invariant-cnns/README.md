## CARPK-only Workspace

Dieses Teilprojekt ist auf das CARPK-Dataset reduziert.

Enthalten sind nur noch:
- CARPK-Daten unter `data/carpk`
- YOLO-Trainingslogik fuer CARPK
- CARPK-SLURM-Beispiele

### Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
```

### Datenstruktur

```text
data/carpk/
  data.yaml
  raw/
    images/
    labels/
    ImageSets/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
    xyxy/
    xywh/
    dots/
```

### Training

```bash
uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 +dataset=carpk \
  ++experiment=carpk_train_size \
  ++dataset.train_size=50,100,500 \
  ++model.batch_size=4 \
  ++model.epochs=100 \
  ++model.patience=6
```

```bash
uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov11 +dataset=carpk \
  ++experiment=carpk_model_size \
  ++model.version=yolo11n.pt,yolo11s.pt \
  ++dataset.train_size=500 \
  ++model.batch_size=4 \
  ++model.epochs=100 \
  ++model.patience=6
```

### MLflow

Lokal:

```bash
mlflow ui --backend-store-uri mlruns
```

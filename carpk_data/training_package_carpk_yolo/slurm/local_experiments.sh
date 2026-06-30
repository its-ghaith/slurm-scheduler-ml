echo "Start experiments"

uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=cnn_base +dataset=carpk ++experiment=train_size_local \
  ++dataset.train_size=50,100 \
  ++dataset.img_size=[512,256] \
  ++remote_mlflow=False \
  ++model.epochs=100 \
  ++model.batch_size=8 \
  ++model.sigma=2 \
  ++model.alpha=100 \
  ++model.lambda_count=0.01

uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 +dataset=foci ++experiment=foci_yolo \
  ++dataset.train_size=158 ++dataset.img_size=660 \
  ++model.epochs=100 ++model.patience=8 \
  ++model.batch_size=8 ++model.flipud=0.3 ++model.fliplr=0.3 \
  ++model.hsv_s=0.4 ++model.translate=0.0

# uv run python countcv/effcv/run_experiment.py \
#   --multirun \
#   +model=yolov8 +dataset=synthcells ++experiment=synth_train_size_local \
#   ++dataset.train_size=50,100,False \
#   ++remote_mlflow=false \
#   ++model.epochs=100 ++model.patience=8 \
#   ++model.batch_size=8 \
#   ++model.degrees=20 ++model.hsv_s=0.7 \
#   ++model.mosaic=0.5 ++dataset.img_size=256

# uv run python countcv/effcv/run_experiment.py \
#   --multirun \
#   +model=yolov8 +dataset=carpk ++experiment=carpk_train_size_yolo_local_iv \
#   ++dataset.train_size=50,100 \
#   ++remote_mlflow=false \
#   ++model.epochs=100 ++model.patience=6 \
#   ++model.batch_size=4 ++dataset.img_size=640 \

# uv run python countcv/effcv/run_experiment.py \
#   --multirun \
#   +model=yolov8 ++model.version=yolov8n.pt,yolov8s.pt +dataset=carpk ++experiment=carpk_model_size_yolo_local_iv \
#   ++dataset.train_size=500 \
#   ++remote_mlflow=false \
#   ++model.epochs=100 ++model.patience=6 \
#   ++model.batch_size=4 ++dataset.img_size=640 \

# uv run python countcv/effcv/run_experiment.py \
#   --multirun \
#   +model=yolov11 ++model.version=yolo11n.pt,yolo11s.pt +dataset=carpk ++experiment=carpk_model_size_yolo_local_iv \
#   ++dataset.train_size=500 \
#   ++remote_mlflow=false \
#   ++model.epochs=100 ++model.patience=6 \
#   ++model.batch_size=4 ++dataset.img_size=640 \

echo "Experiments finished"
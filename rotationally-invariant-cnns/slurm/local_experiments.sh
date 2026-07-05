echo "Start experiments"

uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 +dataset=carpk ++experiment=carpk_train_size_local \
  ++dataset.train_size=50,100,False \
  ++remote_mlflow=false \
  ++model.epochs=100 ++model.patience=6 \
  ++model.batch_size=4 ++dataset.img_size=640

uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 ++model.version=yolov8n.pt,yolov8s.pt +dataset=carpk ++experiment=carpk_model_size_local \
  ++dataset.train_size=500 \
  ++remote_mlflow=false \
  ++model.epochs=100 ++model.patience=6 \
  ++model.batch_size=4 ++dataset.img_size=640

uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov11 ++model.version=yolo11n.pt,yolo11s.pt +dataset=carpk ++experiment=carpk_model_size_local_v11 \
  ++dataset.train_size=500 \
  ++remote_mlflow=false \
  ++model.epochs=100 ++model.patience=6 \
  ++model.batch_size=4 ++dataset.img_size=640

echo "Experiments finished"

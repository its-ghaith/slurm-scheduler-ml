import os

from run_carpk_yolo_pipeline import main


if __name__ == "__main__":
    # Force distributed-style device default for multi-GPU runs.
    os.environ.setdefault("TRAIN_DEVICE", "0,1")
    main()


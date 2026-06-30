### Start GUI

To start the GUI, run:
```
streamlit run src/countcv/focidet/gui/foci_counter.py --server.address localhost
```


### Set up environment
- Clone repository: git clone <URL>  
- python3.12 -m venv .venv
- source venv/bin/activate
- pip install uv
- uv sync
- pre-commit install

Optional: in VSCode, press STG+ALT+P, "Python: select interpreter" and make sure that "./.venv/bin/python" is selected. 

>>> [!note]
The python package `escnn` and in particular `py3nj` requires a Fortran compiler. If `uv sync` fails, this is the likely cause. Follow the instructions in your terminal to fix it and run `uv sync` or `uv pip install` again.
>>> 

### Using LFS (Large File Storage) for image data
The data is managaged by LFS. To set it up in your environment open a terminal inside the project and run
```
sudo apt-get install git-lfs
git lfs install
git pull
# or to force lfs update run
git lfs pull
```

To track new files by LFS (for example all .TIF files) run
```
git lfs track "*.TIF"
```

Settings for Files tracked by LFS are configured in .gitattributes. 
Files must not be ignored by .gitignore to be tracked by LFS.

### data structure as expected by ultralytics
<pre>├── images
│   ├── train
│   └── val
│   └── test
├── labels
│   ├── train
│   ├── val
│   └── test
└── data.yaml 
</pre>
- images as used for training
- labels 
    - in normalized xywh format (0 to 1, x_center y_center width height) 
    - one .txt file per image with same filename

### uc_cells data structure
Original images as received from BfS are in ./data/ultralytics/raw

Images for training are in ./data/uc_cells/images according to ultralytics guideline with bboxes training labels in ./data/uc_cells/labels

To train ultralytics models data.yaml must exist specifying train and validation data. 

In uc_cells/naive_labels labels are stored that have been created by first *naive* approach containing .json files for each image with foci count and xyxy bboxes labels.

intensity_labels.csv in uc_cells/images/ specifies radiation intensity for each image as specified in raw folder structure.

### Synthetic cells data structure
Raw data is downloaded from https://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html and saved in `data\synthetic_cells\raw`. The dot annotations are saved as png image files with single bright pixels where objects are and black pixels everywhere else. To save filesize and allow for easy data augmentation, we save the data as json files in `data\synthetic_cells\labels\dots`. 

### CarPK data set
We requested the data via https://lafi.github.io/LPN/ and it is saved in a raw folder. Preprocessing steps for density maps and ultralytics workflow happen similarly to the other two datasets.

### Set up online MLFlow
- Follow the intructions given here: https://gitlab.ai-env.de/ki-lab/infrastructure/software/authmgr to add a new application.  
- Copy your Application ID and paste it into config/config.yaml.example as client_id. 
- Rename config.yaml.example to config.yaml 
- Run main. In the terminal, you will get a message "Please log into gitlab: <Link>". 
- Click on the link, a browser tab will open to authorize your device to access your gitlab account.  
- A user code is already filled in. Click "Authorize", then "confirm".  

### Offline MLFlow

If you want to test something quickly and locally, it might be sensible to use local MLFlow. For this, change the global variable `REMOTE_MLFLOW=False` in `main.py` and see the results by executing in terminal

`mlflow ui --backend-store-uri mlruns`


## Run experiments (with hydra)
### General
hydra is a framework to run multiple training runs in different configurations. It is used here to run reproducable/comparable experiments with different settings, datasets and models. Configurations are stored in yaml specifications in `./config`. To control the number of runs per parameter set adapt the seeds in defaults.yaml. To start experiments run. Make sure that you include a dataset and model with `+dataset={dataset} +model={model}`. Available options are the names of the yaml files in `./config/{model, dataset}`. Overwrite options with ++, e.g. `++dataset.train_size=50`.
```
uv run python src/countcv/effcv/run_experiment.py --multirun +dataset=foci +model=yolov8,yolov11 dataset.train_size=50,100
# to create a new experiment set a different experiment in defaults.yaml or overwrite parameters inside multirun using ++ e.g.

uv run python src/countcv/effcv/run_experiment.py --multirun +dataset=foci +model=yolov8,yolov11 dataset.train_size=50,100 ++experiment=new_experiment_name

# mlflow run . --env-manager local # does not work with mlflow setup.
```

It is not possible to sweep conditionally. So sweeping over a subset of e.g. models required multiple commands.
### Reproduce experiments
#### Train size experiments object detection
This takes 2-23 minutes per run on laptop gpu.
```
uv run python src/countcv/effcv/run_experiment.py --multirun +model=yolov8 +dataset=foci ++experiment=train_size ++model.batch_size=8 ++dataset.train_size=300,400,500,1000,2000

uv run python src/countcv/effcv/run_experiment.py --multirun +model=yolov8 +dataset=foci ++experiment=train_size ++model.batch_size=4 ++dataset.train_size=50,100,200

# synthethic cells object detection
uv run python src/countcv/effcv/run_experiment.py   --multirun ++experiment=train_size +model=yolov8,yolov11 +dataset=synthcells ++experiment=train_size_synth_cells ++dataset.train_size=50,100  ++model.batch_size=8 ++model.patience=8 ++model.degrees=20.0 ++model.mosaic=0.5
```

#### Good parameter setup for density maps per dataset

A sensible set of hyperparameters for each dataset is shown below for the density map models.
> **_NOTE:_**  model.sigma and img_size are dependent as sigma operates on a pixel level. Therefore one should adjust sigma proportionally to changes in img_size (e.g., if img_size and therefore the side-length of the input image is doubled, sigma should be doubled as well).

```
uv run python src/countcv/effcv/run_experiment.py --multirun +model=cnn_base +dataset=foci ++dataset.img_size=220 ++model.tilesize=224 ++model.sigma=2.25 ++model.beta1=0.83 ++model.beta2=0.85 ++model.degree=22 ++model.flip_rotate=True ++model.infer_normalisation=True ++model.lambda_count=0.0008
```

```
uv run python src/countcv/effcv/run_experiment.py --multirun +model=cnn_base +dataset=synthcells ++dataset.img_size=256 ++model.sigma=2.52 ++model.beta1=0.805 ++model.beta2=0.86 ++model.degree=10 ++model.flip_rotate=True ++model.infer_normalisation=True ++model.lambda_count=0.00013
```

```
uv run python src/countcv/effcv/run_experiment.py --multirun +model=cnn_base +dataset=carpk ++dataset.img_size=[256,128] ++model.sigma=1.61 ++model.beta1=0.86 ++model.beta2=0.87 ++model.degree=13 ++model.flip_rotate=False ++model.infer_normalisation=False ++model.lambda_count=0.0011
```

### Plotting mlflow results 
- plot functions are in effcv/plotting_pipeline.py -> add new functions here for other plots
- Run the following in Terminal for sample plots with shared runs that are stored in *mlruns_shared_no_artifacts*:
```
python -m countcv.effcv.plotting_pipeline --output_dir "/path/to/plots"
# or to use another mlflow runs directory than the one tracked by lfs
python -m countcv.effcv.plotting_pipeline --mlflow_dir /path/to/custom/mlflowruns --output_dir /path/to/plots
```
- or create your own notebook and use the plotting functions as in plotting_pipeline.py main to experiment with other experiments/runs/plots.


### Create bbox labels using zero-shot detection
Run *./effcv/create_zero_shot_bbox_labels.py* to create fake **ground truth** labels for object detection training. Images with bboxes are saved to ./boxed_images to manually check results. The run for all images (~6000) takes >30 minutes on laptop GPU.

Weights are stored in ./models tracked by LFS. Model config is stored in config/


### Inference 
Make sure you have a directory with model checkpoints, e.g. `./checkpoints`. It has to contain the following files:
```
- cnn_base/model.pt
- cnn_base/transform_config.yaml
- yolo/best.pt
```

Then start the inference script:
```
python -m countcv.focidet.main path/to/images path/to/checkpoints
```

# Illumination Estimation and Relighting

This repository contains the source code for Single Indoor Scene Image Illumination Estimation and Relighting. 

## Setup
To setup the required environent to run the experiments, you could do it in two ways.

### Option 1: Using pip
In a new python environment created using either ```conda``` or ```virtualenv```, use the following command to install the required packages in the ```requirements.txt``` file.
```
pip install -r requirements.txt
```

### Option 2: Using conda
This option allows you use conda to setup all the dependencies by using the `environment.yaml` file.
```
conda create -n illum_est
conda activate illum_est
conda env update -f environment.yaml
```

## Dataset

All our experiments use the [Multi-illumination](https://projects.csail.mit.edu/illumination/) dataset for both illumination estimation and relighting. The dataset is split into `train`, `test`, and `val` text files and can be found under `data` folder.

## Training
To train the all illumination estimation models, set the `model` flag in `train.sh` to the appropriate architecture name (i.e., original, vgg, resnet18, or unet).
```
bash train.sh
```

To train the baseline image relighting modele for left-right relighting along with illumination estimation.
```
bash train_relighting_baseline.sh
```

Additionally, to train the baseline random relighting model use `--random_relight` boolean flag in `train_relighting_baseline.sh` script.

To train the best model for relighting.
```
bash train_relighting.sh
```

### Evaluation

To evaluate the model use the appropriate evaluation script. For illumination estimation models:
```
bash eval.sh
```

For image relighting models:
```
bash eval_relighting.sh
```



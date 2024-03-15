# Road-Segmentation-Project
We compare and contrast the recently published Y-Net architecture, a model for segmentation using a large and a small arm, with a standard U-Net and demonstrate that unlike claimed by the Y-Net’s authors there is no apparent advantage to Y-Net by providing a detailed ablation on the number of arms used. Furthermore, we propose so-called Large Upsampling (LU) which improves Y-Net’s performance in image segmentation. Lastly, we provide a first open implementation of the Y-Net architecture, as well as our improvements and proposed changes.

## Environment

The environment uses `pipenv`. This is to ensure we all have the same environment and all use the same library versions (also makes reproducing easy).

Run `pipenv install` to install all dependencies. It requires Python3.8 which should be easy to install (e.g. via ppa:deadsnakes/ppa on Ubuntu).

Enable the virtual environment using `pipenv shell`


## Data Directory

The project supports three datasets:
- the CIL Data (see Kaggle)
- The Cities Dataset (see https://github.com/alpemek/aerial-segmentation/blob/master/dataset/download_dataset.sh)
- Massachusetts Road Dataset

Your data should be organized as follows

```
cil_data_root
 - cil_data (the CIL data as downloaded)
 - ma_data (the massachusetts road dataset)
 - cities_data (the content city folders)
```

You can choose which data to use wih the `--train_data` and `--val_data` flags; they take a list of arguments.
Currently implemented are `cil`, `berlin`, `paris`, `zurich`, `chicago`. If a dataset is in both `--train_data` and
`--val_data` it will be split, otherwise if its in `--train_data` everything will be used for training.

## During Dev Process

Download the Massachusetts Road Dataset and put the path into the configuration dict.

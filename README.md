## Overview

This is a tiny project to practice deep learning skills. The model distinguishes between images with and without Saint George.

I used EfficientNet-B2 with fine-tuning. It achived 93.8% accuracy.

All images are removed from the output for copyright reasons. 

### Python dependencies / Poetry installation
Make sure you have Python3.9 installed and, to install Poetry, just follow the instructions from https://python-poetry.org/docs/#installation

## How to run
The commands below will install the dependencies and open the project.

```
poetry install
poetry shell
jupyter-notebook &
```

The last notebook contains the current version of the project.
## Dataset
You can download dataset from [here](data/geogres.csv) and [here](data/non_georges.csv). It contains 6047 images that are divided into 2 categories: 2681 with St. George and 3366 without him. 

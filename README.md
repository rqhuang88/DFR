

# Introduction

This repository contains the code for the paper [Non-Rigid Shape Registration via Deep Functional Maps Prior](https://deepfunctionalregistration.github.io/)


### This code is under construction. The final version of code will be released soon.


## Installation
To install requirements:

pip install -r requirements.txt

Installing PyTorch may require an ad hoc procedure, depending on your computer settings.


## Dataset
We have uploaded the SCAPE_r dataset. And you can make your own dataset like:
```
SCAPE_r
--shapes_train
--shapes_test
--corres(vts files, if not, you should delete the vts_list in dataset.py)
```

## Training
In the DFM folder, run the following command to train our modified DGCNN model on the train set:
```trian
python train.py
```

## Evaluation
In the registration folder, run the following command to evaluate the trained model on the test set:
```eval
python test.py
```
the results will be saved in the results folder.

### License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

If you use this code, please cite our paper.

```
@inproceedings{NEURIPS2023_b654d615,
 author = {Jiang, Puhua and Sun, Mingze and Huang, Ruqi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {58409--58427},
 publisher = {Curran Associates, Inc.},
 title = {Non-Rigid Shape Registration via Deep Functional Maps Prior},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/b654d6150630a5ba5df7a55621390daf-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

```

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). 
For any commercial uses or derivatives, please contact us.

# Paper
Spatially and Spectrally Consistent Deep Functional Maps  <br/>
[Mingze Sun]<sup>1</sup>, [Shiwei Mao]<sup>1</sup>, [Puhua Jiang]<sup>1,2</sup>,  [Maks Ovsjanikov]<sup>3</sup>, [Ruqi Huang]<sup>1</sup> <br/>
<sup>1 </sup>Tsinghua Shenzhen International Graduate School, China,   <sup>2 </sup>Peng Cheng Laboratory, China,  <br/>
<sup>3 </sup>LIX, Ecole polytechnique, IP Paris, France <br/>
ICCV, 2023 <br/>

# Overview 
In this paper, we formulate a simple yet effective two-branch design of unsupervised DFM based on our theoretical justification, which introduces spatially cycle consistency.

Qualitative visualizations of segmentation results:

<img src="./asset/teaser.png" width="500" height="200"/>

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Datasets
Within each dataset folder, the following structure is expected:

    SCPAE_r/
    ├── shapes_train
    └── shapes_test
    └── corres


# Training
```bash
python train.py --config scape_r
```

# Testing
```bash
python eval.py --config scape_r --model_path ckpt.pth --save_path results_path
```


# Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{sun2023spatially,
  title={Spatially and Spectrally Consistent Deep Functional Maps},
  author={Sun, Mingze and Mao, Shiwei and Jiang, Puhua and Ovsjanikov, Maks and Huang, Ruqi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14497--14507},
  year={2023}
}
```

<div align="center">

# Boosting Visual Reprogramming for CLIP with Dual Granularity Alignment [CVPR26 Highlight]

[Jiayang Wu](https://openreview.net/profile?id=~Jiayang_Wu6)<sup>1</sup> &nbsp;&nbsp;
[Xinyang Chen](https://openreview.net/profile?id=~Xinyang_Chen1)<sup>1†</sup> &nbsp;&nbsp;
[Ke Lv](https://openreview.net/profile?id=~Ke_Lv2)<sup>2,4</sup> &nbsp;&nbsp;
[Weili Guan](https://openreview.net/profile?id=~Weili_Guan4)<sup>1,3†</sup>

<sup>1</sup>Harbin Institute of Technology (Shenzhen) &nbsp;&nbsp;
<sup>2</sup>University of the Chinese Academy of Sciences  
<sup>3</sup>Shenzhen Loop Area Institute &nbsp;&nbsp;
<sup>4</sup>Peng Cheng Laboratory

</div>

## Requirements

Please download the `requirements.txt` file and install the required dependencies before running the project.

## Dataset preparation
Please set `DOWNSTREAM_PATH` in `cfg.py` to the root directory containing all datasets. For example:

```python
DOWNSTREAM_PATH = "/xxx/xxx/xxx/dataset"
```
The **expected directory structure** is shown below:


```text
DOWNSTREAM_PATH/
├── caltech-101/
│   ├── 101_ObjectCategories/
│   ├── Annotations/
│   └── ...
├── dtd/
│   ├── images/
│   │   ├── banded/
│   │   └── ...
│   ├── labels/
│   └── ...
├── eurosat/
│   ├── images/
│   │   ├── AnnualCrop/
│   │   └── ...
│   └── ...
├── fgvc_aircraft/
│   ├── data/
│   │   ├── images/
│   │   ├── families.txt
│   │   └── ...
│   ├── labels/
│   └── ...
├── food-101/
│   ├── images/
│   │   ├── apple_pie/
│   │   └── ...
│   ├── meta/
│   └── ...
├── imagenet/
│   ├── images/
│   │   ├── train/
│   │   │   ├── n01440764/
│   │   │   └── ...
│   │   ├── val/
│   │   │   ├── n01440764/
│   │   │   └── ...
│   │   └── ...
│   ├── test/
│   └── ...
├── oxford_flowers/
│   ├── jpg/
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   └── ...
│   └── ...
├── oxford_pets/
│   ├── images/
│   │   ├── Abyssinian_1.jpg
│   │   └── ...
│   ├── annotations/
│   └── ...
├── resisc45/
│   ├── NWPU-RESISC45/
│   │   ├── airplane/
│   │   └── ...
│   └── ...
├── stanford_cars/
│   ├── train/
│   │   ├── Acura Integra Type R 2001/
│   │   └── ...
│   ├── test/
│   │   ├── Acura Integra Type R 2001/
│   │   └── ...
│   └── ...
├── sun397/
│   ├── SUN397/
│   │   ├── a/
│   │   │   ├── abbey/
│   │   │   └── ...
│   │   ├── b/
│   │   └── ...
│   └── ...
└── ucf101/
    ├── UCF-101-midframes/
    └── ...
````

## Hyper-parameters

| Hyper-parameter         | Aircraft |  UCF | Flowers |   SUN | Resisc |  DTD | Caltech | Cars |  ESAT |  Food |  Pets |  INet |
| ----------------------- | -------: | ---: | ------: | ----: | -----: | ---: | ------: | ---: | ----: | ----: | ----: | ----: |
| Optimized num_sg_layers |        2 |    3 |       2 |     3 |      2 |    2 |       2 |    2 |     2 |     2 |     1 |     2 |
| Optimized num_vg_scales |        1 |    2 |       1 |     2 |      1 |    1 |       1 |    1 |     1 |     2 |     1 |     1 |
| Number of Crops         |        5 |    4 |       5 |     3 |      5 |    5 |       5 |    5 |     5 |     3 |     5 |     3 |
| proj_lr                 |     0.01 | 0.01 |    0.01 | 0.001 |   0.01 | 0.01 |    0.01 | 0.01 | 0.001 | 0.001 | 0.001 | 0.001 |

For the remaining hyper-parameters, the default settings are recommended.

## Training and Test

Example command for the Aircraft dataset to generate hierarchical label structures:
```bash
python plh.py --dataset fgvc --seed 0 --model_name ViT-B/16 --num_sg_layers 2 --num_vg_scales 2
```
Example training command for the Aircraft dataset:
```bash
python dga.py \
  --dataset fgvc \
  --model_name ViT-B/16 \
  --batchsize 64 \
  --num_sg_layers 2 \
  --proj_lr 0.01 \
  --num_vg_scales 2 \
  --n_crops 5
```

Example test command for the Aircraft dataset if you already have a trained checkpoint:
```bash
python dga.py \
  --dataset fgvc \
  --model_name ViT-B/16 \
  --batchsize 64 \
  --num_sg_layers 2 \
  --proj_lr 0.01 \
  --num_vg_scales 2 \
  --n_crops 4 \
  --test
```

## Acknowledgements

This project is built upon and further developed from [AttrVR](https://github.com/tmlr-group/AttrVR) and [DecoupledVP](https://github.com/tmlr-group/DecoupledVP). 

We sincerely thank the authors for open-sourcing their code, which served as an important foundation for our work.

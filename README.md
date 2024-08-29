# HDRFlow: Real-Time HDR Video Reconstruction with Large Motions
### [Project Page](https://openimaginglab.github.io/HDRFlow/) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_HDRFlow_Real-Time_HDR_Video_Reconstruction_with_Large_Motions_CVPR_2024_paper.pdf)) <br>

Gangwei Xu, Yujin Wang, Jinwei Gu, Tianfan Xue, Xin Yang <br>
CVPR 2024 <br><br>

![teaser](docs/static/images/teaser.png)
We propose a robust and efficient flow estimator tailored for real-time HDR video reconstruction, named HDRFlow. HDRFlow predicts HDR-oriented optical flow and exhibits robustness to large motions. We compare our HDR-oriented flow with RAFT's flow. RAFT's flow is sub-optimal for HDR fusion, and alignment may fail in occluded regions, leading to significant ghosting artifacts in the HDR output.<br>
Compared to previous SOTA methods, our HDRFlow enables real-time reconstruction of HDR video from video sequences captured with alternating exposures.

## Installation

### Set up the python environment

```
conda create -n hdrflow python=3.10
conda activate hdrflow
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Set up datasets

#### 0. Set up training datasets
We utilize Vimeo-90K and Sintel datasets as our training datasets. The Vimeo-90K dataset can be downloaded at [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset). The Sintel dataset can be downloaded at [BaiduYun](https://pan.baidu.com/s/1GBRyIWZmlGbTGptYX1j-zQ?pwd=gnqj). The training datasets are organized as follows:
```
├── HDRFlow/data
            ├── vimeo_septuplet
                ├── sequences
            ├── Sintel
                ├── training
                    ├── clean
                    ├── final
                    ├── flow
                    ├── reverse_flow
                    ├── flow_2
                    ├── reverse_flow_2
```

#### 1. Set up test datasets
We evaluate our method on HDR_Synthetic_Test_Dataset (Cinematic Video dataset), DeepHDRVideo, and TOG13_Dynamic_Dataset (HDRVideo dataset). These datasets can be downloaded at [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset). The HDR_Synthetic_Test_Dataset contains two synthetic videos (POKER FULLSHOT and CAROUSEL FIREWORKS), DeepHDRVideo consists of both real-world dynamic scenes and static scenes that have been augmented with random global motion. The TOG13_Dynamic_Dataset does not have ground truth, so we use it for qualitative evaluation. The test datasets are organized as follows:

```
├── HDRFlow/data
            ├── HDR_Synthetic_Test_Dataset
            ├── dynamic_RGB_data_2exp_release
            ├── static_RGB_data_2exp_rand_motion_release
            ├── dynamic_RGB_data_3exp_release
            ├── static_RGB_data_3exp_rand_motion_release
            ├── TOG13_Dynamic_Dataset
```

## Evaluation and Training
### Demo
You can demo a pre-trained model on ThrowingTowel-2Exp-3Stop from TOG13_Dynamic_Dataset. The TOG13_Dynamic_Dataset can be downloaded at [BaiduYun](https://pan.baidu.com/s/1GBRyIWZmlGbTGptYX1j-zQ?pwd=gnqj).
```
python test_tog13_2E.py
```

### Evaluation
2 Exposures
```
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/dynamic_RGB_data_2exp_release
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/static_RGB_data_2exp_rand_motion_release
python test_2E.py --dataset CinematicVideo --dataset_dir data/HDR_Synthetic_Test_Dataset
python test_tog13_2E.py
```

3 Exposures
```
python test_3E.py --dataset DeepHDRVideo --dataset_dir data/dynamic_RGB_data_3exp_release
python test_3E.py --dataset DeepHDRVideo --dataset_dir data/static_RGB_data_3exp_rand_motion_release
python test_3E.py --dataset CinematicVideo --dataset_dir data/HDR_Synthetic_Test_Dataset
python test_tog13_3E.py
```
### Training
2 Exposures
```
python train_2E.py
```

3 Exposures
```
python train_3E.py
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{xu2024hdrflow,
  title={HDRFlow: Real-Time HDR Video Reconstruction with Large Motions},
  author={Xu, Gangwei and Wang, Yujin and Gu, Jinwei and Xue, Tianfan and Yang, Xin},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgement
This project is based on [DeepHDRVideo](https://github.com/guanyingc/DeepHDRVideo), we thank the original authors for their excellent work.


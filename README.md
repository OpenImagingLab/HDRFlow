# HDRFlow: Real-Time HDR Video Reconstruction with Large Motions
### [Project Page](https://openimaginglab.github.io/HDRFlow/) | [Video]() | [Paper]() | [Data]() <br>

Gangwei Xu, Yujin Wang, Jinwei Gu, Tianfan Xue, Xin Yang <br>
CVPR 2024 <br><br>

![teaser](docs/static/images/teaser3.png)
We propose a robust and efficient flow estimator tailored for real-time HDR video reconstruction, named HDRFlow. HDRFlow predicts HDR-oriented optical flow and exhibits robustness to large motions. We compare our HDR-oriented flow with RAFT's flow. RAFT's flow is sub-optimal for HDR fusion, and alignment may fail in occluded regions, leading to significant ghosting artifacts in the HDR output.<br>
Compared to previous SOTA methods, our HDRFlow enables real-time reconstruction of HDR video from video sequences captured with alternating exposures.

- [ ] Release the training and testing code.

## Installation

### Set up the python environment

```
conda create -n hdrflow python=3.10
conda activate hdrflow
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### Set up datasets

#### 0. Set up training datasets

```
├── /data
    ├── vimeo_septuplet
        ├── sequences
    ├── Sintel
        ├── clean
        ├── final
        ├── flow
        ├── reverse_flow
        ├── flow_2
        ├── reverse_flow_2
    ├── HDR_Synthetic_Test_Dataset
    ├── dynamic_RGB_data_2exp_release
    ├── static_RGB_data_2exp_rand_motion_release
    ├── dynamic_RGB_data_3exp_release
    ├── static_RGB_data_3exp_rand_motion_release
    ├── TOG13_Dynamic_Dataset
```



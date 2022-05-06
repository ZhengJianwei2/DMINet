# CTFI-Net

## Papers
* Building Change Detection in Bi-temporal Remote Sensing Images using Cross-temporal Feature Interaction Network (TGARS Submitted)

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and two GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6.

## 2. Download the datesets:
* LEVIR-CD:
[LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* WHU-CD:
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
* GZ-CD:
[GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

and put them into data directory.

## 3. Download the models (loading models):

* [models]() code: 

and put them into checkpoints directory.

## 4. Evaluate

    python demo.py


# DMINet

## Papers
* Change Detection on Bi-temporal Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network (TGARS Submitted)

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and two GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6.

## 2. Download the datesets:
* LEVIR-CD:
[LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* WHU-CD:
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
* GZ-CD:
[GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)
* SYSU-CD:
[SYSU-CD](https://github.com/liumency/SYSU-CD)

and put them into data directory.

## 3. Download the models (loading models):

* [models](https://pan.baidu.com/s/1m2vDFG8-FOCYdeidLeYOsA) code: x1qk

and put them into checkpoints directory.

## 4. Train

    python main_cd.py
    
## 5. Test

    python eval_cd.py


# ICIF-Net

## Papers
* ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection (TGARS 2022) [paper](https://ieeexplore.ieee.org/document/9759285) and [source_code](https://github.com/ZhengJianwei2/ICIF-Net/)

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and two GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6. Please install related libraries before running this code:

    pip install -r requirements.txt

## 2. Download the datesets:
* LEVIR-CD:
[LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* LEVIR-CD+:
[LEVIR-CD+](https://github.com/S2Looking/Dataset)
* WHU-CD:
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
* GZ-CD:
[GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)
* SYSU-CD:
[SYSU-CD](https://github.com/liumency/SYSU-CD)

and put them into data directory.

## 3. Download the models (loading models):

* [models](https://pan.baidu.com/s/1XmE1T9G_NgTXVuZdLmaVhg ) code: pmgt

and put them into checkpoints directory.

## 5. Test

    python main_cd.py

## 5. Test

    python eval_cd.py

## 6. Cite
If you use ICIF-Net in your work please cite our paper:
* BibTex：


    @ARTICLE{9759285,
      author={Feng, Yuchao and Xu, Honghui and Jiang, Jiawei and Liu, Hao and Zheng, Jianwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection}, 
      year={2022},
      volume={60},
      number={},
      pages={1-13},
      doi={10.1109/TGRS.2022.3168331}
    }

* Plane Text：
	
    Y. Feng, H. Xu, J. Jiang, H. Liu and J. Zheng, "ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 4410213, doi: 10.1109/TGRS.2022.3168331.
    
    

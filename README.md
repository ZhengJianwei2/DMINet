# DMINet

## Papers
* Change Detection on Bi-temporal Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network (TGARS 2023) [paper](https://ieeexplore.ieee.org/document/10034787) and [source_code](https://github.com/ZhengJianwei2/DMINet/)
* ICIF-Net: Intra-Scale Cross-Interaction and Inter-Scale Feature Fusion Network for Bitemporal Remote Sensing Images Change Detection (TGARS 2022) [paper](https://ieeexplore.ieee.org/document/9759285) and [source_code](https://github.com/ZhengJianwei2/ICIF-Net/)

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6.

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


## 6. Cite
If you use DMINet in your work please cite our paper:
* BibTex：


    @ARTICLE{10034787,
      author={Feng, Yuchao and Jiang, Jiawei and Xu, Honghui and Zheng, Jianwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Change Detection on Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network}, 
      year={2023},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TGRS.2023.3241257}
    }

* Plane Text：
	
    Y. Feng, J. Jiang, H. Xu and J. Zheng, "Change Detection on Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3241257.
    
    
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
    
    

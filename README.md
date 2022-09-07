# A Superpixel-Guided Unsupervised Fast Semantic Segmentation Method of Remote Sensing Images

English | [简体中文](./README-zh_CN.md) 

This repo explains the working of our Unsupervised Segemntaion Algorithm published in IEEE Geoscience and Remote Sensing Letters. Our method can perform GPU acceleration computing, thus greatly improving the efficiency.

You can access the [**_full paper here_**](https://ieeexplore.ieee.org/document/9854897?source=authoralert).

## Abstract
Semantic segmentation is one of the fundamental tasks of pixel-level remote sensing image analysis. Currently, most high-performance semantic segmentation methods are trained in a supervised learning manner. These methods require a large number of image labels as support, but manual annotations are difficult to obtain. To address the problem, we propose an efficient unsupervised remote sensing image segmentation method based on superpixel segmentation and fully convolutional networks (FCNs) in this letter. Our method can achieve pixel-level images segmentation of various scales rapidly without any manual labels or prior knowledge. We use the superpixel segmentation results as synthetic ground truth to guide the gradient descent direction during FCN training. In experiments, our method achieved high performance compared with current unsupervised image segmentation methods on three public datasets. Specifically, our method achieves an adjusted mutual information (AMI) score of 0.2955 on the Gaofen Image Dataset (GID), while processing each image of size 7200 × 6800 pixels in just 30s.

### Network Architecture
Following images shows the Complete Network Architecture.

![alt text](figures/model.png)

### Example Output
Original images and the segmentation results of different segmentation methods on the EvLab-SS, the GID and the ISPRSV datasets: (1) original image, (2) ground truth, (3) our proposed method, (4) ISODATA, (5)  K-means

![alt text](figures/result.png)

## Getting Started
### Requement
```
python=3.7.0
pytorch=1.8.1
gdal=3.4.0 
scikit-image=0.18.1
scikit-learn=0.24.2
tqdm=4.61.2
```

### Installation
Clone this repo:
```
git clone https://git.chenguanzhou.com/chenguanzhou123/Unsupervised-OBIC-Pytorch.git
cd Unsupervised-OBIC-Pytorch
```
### Run the code
To run our unsupervised segmentation code, you need run the segmentation.py first.
```
python segmentation.py 
python train_net.py -i data/GID_example/example.tif 
```
## File Directory Description
```
filetree 
├── /data/
│  ├── /GID_example/
│  │  └── example.tif
├── /figures/
│  ├── model.png
│  └── result.png
├── /losses/
│  ├── focal_loss.py
│  └── lovasz_losses.py
├── /results/
├── evaluate_gt.py
├── inference.py
├── model.py
├── README.md
├── segmentation.py
├── to_ave_color.py
└── train_net.py

```

## Citation
If you like to use our work, please consider citing us:
```
@article{chen2022superpixel,
  title={A Superpixel-guided Unsupervised Fast Semantic Segmentation Method of Remote Sensing Images},
  author={Chen, Guanzhou and He, Chanjuan and Wang, Tong and Zhu, Kun and Liao, Puyun and Zhang, Xiaodong},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2022},
  publisher={IEEE}
}
```

## License
Code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

## Reference
https://github.com/Yonv1943/Unsupervised-Segmentation

Asako Kanezaki.
**Unsupervised Image Segmentation by Backpropagation.** 
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2018.
([pdf](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf))
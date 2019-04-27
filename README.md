# RealM: Real-time Multi-Person Pose Estimation network with low computational complexity

## Acknowledge
Code structure from Gtihub Repo: [https://github.com/YangZeyu95/unofficial-implement-of-openpose]. Thanks to Zeyu's work!
This is our EECS 598 deep learn final project, for study and research use only. 

## Author
Shiyu Wang, Peng Xue, Xinyu Gao, Siyuan Xie

## Configuration
### Dataset
Download COCO 2017 dataset from their official website [http://cocodataset.org/#download]
Place downloaded feature in the following directory (not exist):
-COCO
    -images
        -train
        -val
    -annotations

### Necessary Packages
via pip: requests, cv2, tensorflow, tensorpack, pycocotools, matplotlib

Also, you need to install swig:
on Linux: '''apt-get install swig'''
on Mac: '''brew install swig'''

Install post-processing module:
In pafprocess folder: 
'''swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace'''

### Data Augmentation
'''python pose_dataset.py'''

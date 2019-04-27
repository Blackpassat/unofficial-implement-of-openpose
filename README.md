# RealM: Real-time Multi-Person Pose Estimation network with low computational complexity

## Acknowledge
Code structure from Gtihub Repo [here]: https://github.com/YangZeyu95/unofficial-implement-of-openpose]. Thanks to Zeyu's work!

This is our EECS 598 deep learn final project, for study and research use only. 

## Author
Shiyu Wang, Peng Xue, Xinyu Gao, Siyuan Xie

## Configuration
### Dataset
Download COCO 2017 dataset from their official website [COCO download]: http://cocodataset.org/#download]

Place downloaded feature in the following directory (not exist):
```
-COCO/
  -images/
    -train/
    -val/
  -annotations/
```

### Necessary Packages
via pip: requests, cv2, tensorflow, tensorpack, pycocotools, matplotlib

Also, you need to install swig:

on Linux: `apt-get install swig`

on Mac: `brew install swig`

Install post-processing module:

In pafprocess folder: 

`swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace`

### Data Augmentation
`python pose_dataset.py`

### Pre-train model download
Using the following folder directory
```
-checkpoints
  -vgg # vgg19 pretrain model download [here]: http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
  -mobilenet # mobilenet model download [here]: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_96.tgz
  -train # your training directory
```

## Code
### Code overview
We have five models, original openpose implementation(\*_vgg), changed feature extractor (\*_mobilenetv2_unchanged_conv), all separable convolution built (\*_), 3 by 3 convolution block built (\*_mn_sepconv33) and final RealM model (\*_mn_sepconv34).

Meanwhile, cpm_\* model for model definition, eval_\* for model evaluation, train_\* for model training.

Note that for this project, due to limited resource, we only manage to train the original openpose implementation and the final RealM model.

### Instructions
To train:
`python train_[SPECIFY MODEL]`

To evaluate trained model
`python eval_[SPECIFY MODEL]`

To perform a live demo
`python run.py`

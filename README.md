# Multi-Loss Weighting with Coefficient of Variations
This is the repository for our WACV 2021 work **Multi-Loss Weighting with Coefficient of Variations**. \
[WACV](https://openaccess.thecvf.com/content/WACV2021/papers/Groenendijk_Multi-Loss_Weighting_With_Coefficient_of_Variations_WACV_2021_paper.pdf) \
[arXiv](https://arxiv.org/abs/2009.01717)

This repository provides code for trying out CoV-Weighting yourself on depth datasets (KITTI and CityScapes). Feel free to use or change any method in [methods](methods/) and apply them to your own task or dataset.

## Dependencies
A [requirements](requirements.txt) file is available to retrieve all dependencies. Create a new python environment and install using:
```shell
pip install -r requirements.txt
``` 

## Training
Models can be trained by specifying your data directory, a model name and any method. For example:
```shell
python train.py --data_dir data/ --model_name [MODEL_NAME] --method cov-weighting
```
There are many, many options for training the models. Have a look at the [options](options/).
Before training, a directory is created in [saved_models](saved_models/) including a text file with all training parameters. After training, models are stored in this directory as well.

## Testing  
All testing code is limited to testing on the Eigen split of the KITTI dataset and can be used as such:
```shell
python test.py --data_dir data/ --model_name [MODEL_NAME]
```

## Data
This work has been trained on rectified stereo pairs. For this two datasets have been used: KITTI and CityScapes.
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
In this work the split of **eigen** is used to train and test model. This set contains 22600 training images, 888 validation images and 652 test images.  
In the [filenames](utils/filenames) folder there are lists that detail which images correspond to which set. All data can be downloaded by running:
```shell
wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/
```

### [CityScapes](https://www.cityscapes-dataset.com)
To access data of the CityScapes dataset, one has to register an account and then request special access to the ground truth disparities.  
When this data is retrieved the following directories should be put in the [data](data/) folder:  
cs_camera/ with all camera parameters.  
cs_disparity/ with all ground truth disparities.  
cs_leftImg8bit/ with all left images.  
cs_rightImg8bit/ with all right images.

It is good to note that the code assumes all image files are saved in .png format. If you have saved them in any other format, change **IMAGE_EXTENSION** in [config.py](config.py). 

Also, say you wanted to train on a random subset of data, just to see how everything works, use the **--train_ratio** argument as any number in the range [0, 1].

## Results
Results are available upon request.

## References
We implement a few other loss weighting methods as baselines. These works are:

[GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks](http://proceedings.mlr.press/v80/chen18a.html)  
[Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)  
[Multi-task learning as multi-objective optimization](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization)

Besides these works, evaluation and data loading code from [MonoDepth-V2](https://github.com/nianticlabs/monodepth2) was used.

## Citation
If this work was useful for your research, please consider citing:
```
@inproceedings{groenendijk2020multi,
  title={Multi-Loss Weighting with Coefficient of Variations},
  author={Groenendijk, Rick and Karaoglu, Sezer and Gevers, Theo and Mensink, Thomas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1469--1478},
  year={2020}
}
```

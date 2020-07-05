

# There are so many large files in .git folder and I hate them, so I move the code to [https://github.com/qixuxiang/orb-slam2_with_semantic_labelling](https://github.com/qixuxiang/orb-slam2_with_semantic_labelling)

# orb-slam2_with_semantic_label

**Authors:** Xuxiang Qi(qixuxiang16@nudt.edu.cn),Shaowu Yang(shaowu.yang@nudt.edu.cn),Yuejin Yan(nudtyyj@nudt.edu.cn)

**Current version:** 1.0.0

* Note: This repository is mainly built upon [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/pjreddie/darknet/). Many thanks for their great work.

## 0.introduction

**orb-slam2_with_semantic_label** is a  visual SLAM system based on  **[ORB_SLAM2[1-2]](https://github.com/raulmur/ORB_SLAM2)**.
The ORB-SLAM2 is a great visual SLAM method that has been popularly applied in  robot applications. However, this method cannot provide semantic information in environmental mapping.In this work,we present a method to build a 3D dense semantic map,which utilize both 2D image labels from **[YOLOv3[3]](https://github.com/qixuxiang/YOLOv3_SpringEdition)** and 3D geometric information.

![image](https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/png/framwork.png)

## 1. Related Publications

**[Deep Learning Based Semantic Labelling of 3D Point Cloud in Visual SLAM](https://www.researchgate.net/publication/328005677_Deep_Learning_Based_Semantic_Labelling_of_3D_Point_Cloud_in_Visual_SLAM)**


## 2. Prerequisites

### 2.1 requirements
  * Ubuntu 14.04/Ubuntu 16.04/Ubuntu 18.04

  * ORB-SLAM2 

  * CUDA

  * GCC >= 5.0

  * cmake

  * OpenCV

  * PCL1.7 or PCL1.8, may not work with PCL1.9
  
  * libTorch 1.4

    **PS**:(Ubuntu18.04 CUDA10.1 opencv3.4 Eigen3.2.10 PCL1.8 has tested successfully)


### 2.2 Installation

Refer to the corresponding original repositories ([ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/qixuxiang/YOLOv3_SpringEdition) for installation tutorial).

### 2.3 Build 

```
git clone https://github.com/qixuxiang/orb-slam2_with_semantic_label.git

sh build.sh
```


## 3. Run the code
1. Download  `yolov3.weights`, `yolov3.cfg` and `coco.names` from [darknet](https://pjreddie.com/darknet/yolo/) and put them in `bin` folder. Also, these files can be found in [YOLO V3](https://github.com/qixuxiang/YOLOv3_SpringEdition).Then, you should make a dir named `img` in  `bin` folder, that is, you should execute command `sudo mkdir img` in `bin` folder. you can use [libtorch-yolov3](https://github.com/walktree/libtorch-yolov3) replace libYOLOv3SE, see details https://blog.csdn.net/TM431700/article/details/105889614).

2. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it to `data` folder.

3. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in `Examples/RGB-D/associations/`. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```


4. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.You can run the project by:

```
cd bin
./rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM2.yaml ../data/rgbd-data ../data/rgbd-data/associations.txt
```

![image](https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/png/result.png)
## update
1. update 20200705: fix segment fault, make system run faster and use libtorch, thanks for [vayneli](https://github.com/vayneli)!

## Reference
[1] Mur-Artal R, Montiel J M M, Tardos J D. ORB-SLAM: a versatile and accurate monocular SLAM system[J]. IEEE Transactions on Robotics, 2015, 31(5): 1147-1163.

[2] Mur-Artal R, Tardos J D. ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras[J]. arXiv preprint arXiv:1610.06475, 2016.

[3] Redmon, Joseph, and A. Farhadi. "YOLOv3: An Incremental Improvement." (2018).

## License
Our system is released under a [GPLv3 license](https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/License-gpl.txt).

If you want to use code for commercial purposes, please contact the authors.

## Other issue
- We do not test the code there on ROS bridge/node.The system relies on an extremely fast and tight coupling between the mapping and tracking on the GPU, which I don't believe ROS supports natively in terms of message passing.
- I only test the code on OpenCV2 + CDUA8 + CUDNN7 + PCL1.8, and CUDA9/10  will cause segmentation fault.
- Welcome to submit any issue if you have problems, and add your software and computer system information details, such as Ubuntu 16/14,OpenCV 2/3, CUDA 9.0, GCC5.4,etc..

- We provide a [video](http://v.youku.com/v_show/id_XMzYyOTMyODM2OA==.html?spm=a2h3j.8428770.3416059.1) here.

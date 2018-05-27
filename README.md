
# orb-slam2_with_semantic_label

**Authors:** Xuxiang Qi(qixuxiang16@nudt.edu.cn),Shaowu Yang(shaowu.yang@nudt.edu.cn),Yuejin Yan(nudtyyj@nudt.edu.cn)

**Current version:** 1.0.0

* Note: This repository is mainly built upon [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/pjreddie/darknet/). Many thanks for their great work.

## introduction

**orb-slam2_with_semantic_label** is a  visual SLAM system based on  **[ORB_SLAM2[1-2]](https://github.com/raulmur/ORB_SLAM2)**.
The ORB-SLAM2 is a great visual SLAM method that has been popularly applied in  robot applications. However, this method cannot provide semantic information in environmental mapping.In this work,we present a method to build a 3D dense semantic map,which utilize both 2D image labels from **[YOLOv3[3]](https://github.com/raulmur/ORB_SLAM2)** and 3D geometric information.




## 0. Related Publications

**coming soon...**

## 1. Prerequisites

### 1.0 requirements
  * Ubuntu 14.04/Ubuntu 16.04
  * ORBSLAM2 
  * CUDA >=6.5
  * C++11(must)
  * GCC5(must)
  * cmake


### 1.1 Installation

Refer to the corresponding original repositories ([ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/qixuxiang/YOLOv3_SpringEdition) for installation tutorial).



### 2.1 Build 

You should follow the instructions provided by ORB_SLAM2 build its dependencies, we do not list here.
You also need to install NVIDIA and cuda to accelerate it.


### 2.2 run 
1. Download  yolov3.weights, yolov3.cfg and coco.names and put them to bin folder,they can be found in [YOLO V3](https://github.com/qixuxiang/YOLOv3_SpringEdition).

2. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it to data folder.
3. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```


4. Execute the following c Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.My command is :

```
cd bin
./rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM2.yaml ../data/rgbd-data ../data/rgbd-data/associations.txt

```

## Reference
[1] Mur-Artal R, Montiel J M M, Tardos J D. ORB-SLAM: a versatile and accurate monocular SLAM system[J]. IEEE Transactions on Robotics, 2015, 31(5): 1147-1163.

[2] Mur-Artal R, Tardos J D. ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras[J]. arXiv preprint arXiv:1610.06475, 2016.

[3]Redmon, Joseph, and A. Farhadi. "YOLOv3: An Incremental Improvement." (2018).

## License
Our system is released under a [GPLv3 license](https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/License-gpl.txt).

If you want to use code for commercial purposes, please contact the authors.

## Other issue
we do not test the code there on ROS bridge/node.The system relies on an extremely fast and tight coupling between the mapping and tracking on the GPU, which I don't believe ROS supports natively in terms of message passing.

We provide a [video] (https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/License-gpl.txt) here.

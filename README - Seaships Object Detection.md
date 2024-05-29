# Seaships Object Detection

- Seaships数据集是一个包含了7000张照片，6种船分类的数据。


- 该项目是在该数据集上，完成对行船目标的检测、追踪的展示。

- 其追踪文件该项目下无id，因此该数据集目前无法完成目标跟踪。只能通过图像的检测算法，手动导出。

- **本README文档是一个项目概述，其详细方法参考项目子文件夹下的各个README文件。**

由于考核时间提前的原因，该项目在对比测试中以及检测模块的创新点上还有待提高。

## 一、目标检测实现方法

### 1、one-stage（yolo）

##### 1、yolov5-5.0

详细方法请参考  [Readme关于yolov5-5.0的实现.md](Readme%B9%D8%D3%DAyolov5-5.0%B5%C4%CA%B5%CF%D6.md)笔记。 SeashipsDetection/README - Seaships Object Detection.md

其中包含了实现过程以及结果展示。

##### 2、yolov8

详细方法请参考   [README-YOLOV8实现.md](README-YOLOV8实现.md) 笔记。  SeashipsDetection/README-YOLOV8实现.md

在yolov8中添加注意力机制模块。

其中包含了实现过程以及结果展示。

yolov8结果优于yolov5。

##### 3、yolov10

作者在完成项目期间，即5月23日时，有幸阅读了清华大学新发布的yolov10论文。

​	论文名：YOLOv10: Real-Time End-to-End Object Detection

​	地址：[[2405.14458\] YOLOv10: Real-Time End-to-End Object Detection (arxiv.org)](https://arxiv.org/abs/2405.14458)

该文章成功地实现了具有不同模型尺度的新型实时端到端检测器系列，即 YOLOv10-N / S / M / B / L / X。但是由于目前船舶检测数据集不是视频数据，且考虑到yolov10&&yolov5在工业上的稳定性，作者暂不考虑对该数据集使用这一全新的算法。

### 2、two-stage (fasterrcnn)

不同于one-stage的算法，对于目标检测，同样可以采取两步走的算法，其代表为RCNN系列。

由于yolo算法近期的快速发展，其在性能和速度上基本超越了传统CNN算法，因此本来是打算放在最后来尝试Faster-RCNN的算法测试。但目前赶时间，可惜无法完成。




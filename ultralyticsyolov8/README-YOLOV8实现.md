# Yolov8实现Seaships目标检测

### 一、实现

采用yolov8n的模型，训练200轮（yolov5训练300次，在yolov8上减少一些）

Seaships数据集包含7000张照片，其中的data划分按照yolov5的形式（直接复制过来的）。

<img src="C:\Users\李昌峻\AppData\Roaming\Typora\typora-user-images\image-20240526075257065.png" alt="image-20240526075257065" />

<img src="C:\Users\李昌峻\AppData\Roaming\Typora\typora-user-images\image-20240526075315321.png" alt="image-20240526075315321" />

### 二、日志

#### 1、模型

在网上https://github.com/ultralytics/ultralytics找到，download zip。

在官方README文件下找到yolov8n的权重，下载。

#### 2、数据集

同 [yolov5-5.0](..\yolov5-5.0) 文件下的 [VOCdevkit](VOCdevkit) 文件（直接复制就行）

#### 3、配置文件

在路径 SeashipsDetection/ultralyticsyolov8/ultralytics/cfg/datasets/myseaships.yaml 新建yaml文件。

更改其中的配置参数

```python
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../VOCdevkit # dataset root dir
train: images/train # train images (relative to 'path') 128 images
val: images/val # val images (relative to 'path') 128 images
#test: # test images (optional)

# Classes
names:
  0: ore carrier
  1: passenger ship
  2: container ship
  3: bulk cargo carrier
  4: general cargo ship
  5: fishing boat
```

在路径 SeashipsDetection/ultralyticsyolov8/ultralytics/cfg/models/v8/中新建  seashipsyolov8.yaml 文件。

修改

```python
# Parameters
nc: 6 # number of classes
```

#### 4、训练

参考https://docs.ultralytics.com/usage/python/给出的文档。

新建一个py文件 [mytrain.py](mytrain.py) 

#### 5、数据情况

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\labels_correlogram.jpg" alt="labels_correlogram" style="zoom:25%;" /><img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\labels.jpg" alt="labels" style="zoom: 25%;" />

### 三、结果

#### 1、confusion_matrix.png

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\confusion_matrix.png" alt="confusion_matrix" />

归一化：
<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\confusion_matrix_normalized.png" alt="confusion_matrix_normalized" />

#### 2、F1_curve.png

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\F1_curve.png" alt="F1_curve" />

#### 3、PR_curve.png

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\PR_curve.png" alt="PR_curve" />

#### 4、R_curve

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\R_curve.png" alt="R_curve" />

#### 5、P_curve

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\P_curve.png" alt="P_curve" />

#### 6、results

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\results.png" alt="results" />
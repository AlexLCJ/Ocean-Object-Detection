# Yolov8实现Seaships目标检测

前半部分所展示未添加注意力机制的目标检测。该项目在后面添加了注意力机制以用来更好地聚焦重要物体。

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

新建一个py文件 [mytrain.py](mytrain.py) ，通过阅读官方文档，完成训练脚本。

#### 5、数据情况

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\labels_correlogram.jpg" alt="labels_correlogram" style="zoom:25%;" /><img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\labels.jpg" alt="labels" style="zoom: 25%;" />

### 三、结果

#### 1、confusion_matrix.png

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train5\confusion_matrix.png" alt="confusion_matrix"  />

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

### 四、创新方式

#### 1、添加注意力机制'CBAM'

​	CBAM：混合注意力机制。相较于SENET通道注意力机制，他在保留原有通道注意力机制的基础上加入了空间注意力机制，从通道和空间两个方面对网络进行优化，使得优化后的网络可以从通道和空间两个角度获取更为有效的特征，进一步提高模型同时在通道和空间两个角度的特征提取效果，结构图如下图所示：

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\img\20210310210027842.png" />

​	其具体代码情况：

```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
```

```python
def __init__(self, c1, kernel_size=7):
    """Initialize CBAM with given input channel (c1) and kernel size."""
    super().__init__()
    self.channel_attention = ChannelAttention(c1)
    self.spatial_attention = SpatialAttention(kernel_size)

def forward(self, x):
    """Applies the forward pass through C1 module."""
    return self.spatial_attention(self.channel_attention(x))
```

##### 1、修改

###### 1、步骤一

对conv.py、__init__.py、查看，确保已添加。

在 SeashipsDetection/ultralyticsyolov8/ultralytics/nn/tasks.py 中添加CBAM

```python
elif m in {CBAM}:
    c1,c2=ch[f],args[0]
    if c2!=nc:
        c2=make_divisible(min(c2,max_channels)*width,8)
    args = [c1,*args[1:]]
```

###### 2、步骤二

新建yaml参数文件

对backbone区域根据相应情况修改和添加注意力机制的位置

```python
# 添加新的注意力机制
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, CBAM, [256,7]]  # 添加注意力机制
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# head 插入注意力机制
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]] # cat backbone P4   修改concat
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

修改后测试：

```ABAP
my_model_test.py::test_model_forward PASSED  
```

 通过。

###### 3、步骤三

train（在last的权重上测试（时间关系来不及了））

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\img\屏幕截图 2024-05-29 164741.png" />

可以看到CBAM注意力机制已经添加成功。

###### 4、测试

在上一轮last的权重上继续训练200 epochs。

训练结束：得到结果

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\img\屏幕截图 2024-05-29 220816.png" />

其余指标：

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\confusion_matrix.png"  />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\confusion_matrix_normalized.png"  />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\F1_curve.png" />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\P_curve.png" />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\PR_curve.png" />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\R_curve.png" />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\val_batch0_pred.jpg" />

<img src="E:\Ocean Object Detection\OceanObjectDetection\SeashipsDetection\ultralyticsyolov8\runs\detect\train542\val_batch1_pred.jpg" />

“炼丹”效果一般。

#### 2、融合注意力机制

当然可以去github或者官方文档搜寻其他的注意力机制的添加代码，然后炼丹。此处时间紧迫，就不炼丹了。

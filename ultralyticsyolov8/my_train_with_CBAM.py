import multiprocessing
from ultralytics import YOLO


def main():
    # 假设您已经有一个加载了预训练权重的模型实例
    model = YOLO("seaships_CBAM.yaml")  # 使用您的自定义配置文件，而不是从头开始
 # 加载预训练权重（如果可用）

    # 训练模型，并设置 epochs
    results = model.train(
        data="E:\\Ocean Object Detection\\OceanObjectDetection\\SeashipsDetection\\ultralyticsyolov8\\ultralytics\\cfg\\datasets\\myseaships.yaml",
        epochs=100)

    # 评估模型在验证集上的性能
    results_val = model.val()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # 这一行在 Windows 上是必需的
    main()
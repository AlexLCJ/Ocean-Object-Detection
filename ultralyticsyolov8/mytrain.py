import multiprocessing
from ultralytics import YOLO

def main():
    # Create a new YOLO model from scratch
    model = YOLO("seashipsyolov8.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # train
    results = model.train(data="myseaships.yaml", epochs=100)

    # Evaluate the model's performance on the validation set
    results = model.val()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 这一行在 Windows 上是必需的
    main()

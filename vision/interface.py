from detector import QRDetector

detector = QRDetector("yolov8n.pt")
detector.train(
    data_yaml="data/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="auto"
)
from ultralytics import YOLO

video_path = "./data/walking_dogs.mp4"
model = YOLO('yolov8n.pt')


results = model.track(source=video_path,          # For webcam stream as input, change source to 0
                      conf=0.5,                   # Confidence threshold for detection
                      show=True,
                      save=False,
                      tracker="bytetrack.yaml")   # Available trackers; ByteTrack (bytetrack.yaml), BoT-SORT (botsort.yaml)

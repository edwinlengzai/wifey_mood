# src/utils/object_detection_client.py
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetectionClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ObjectDetectionClient, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.model = YOLO("yolo11n.pt")
            self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
            self.class_names = ["person"]  # Add other class names if needed
            self.initialized = True

    def detect_and_track(self, image_data):
        """
        Detect and track objects in the provided image data.

        Args:
            image_data (numpy.ndarray): The image data in numpy array format.

        Returns:
            list: The detection and tracking results.
        """
        detections = self.model(image_data)
        bboxes = []
        for detection in detections:
            for box in detection.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf.item()  # Convert tensor to float
                class_id = int(box.cls[0].item())  # Convert tensor to integer
                if class_id == 0:  # Assuming class_id 0 is for 'person'
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    bboxes.append((bbox, confidence, "person"))

        tracks = self.tracker.update_tracks(bboxes, frame=image_data)
        return tracks

    def draw_tracks(self, image_data, tracks):
        """
        Draw bounding boxes and IDs on the image for each track.

        Args:
            image_data (numpy.ndarray): The image data in numpy array format.
            tracks (list): The tracking results.

        Returns:
            numpy.ndarray: The image with bounding boxes and IDs drawn.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_data, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image_data
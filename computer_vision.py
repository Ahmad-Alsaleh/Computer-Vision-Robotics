from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import cv2


@dataclass
class Point:
    x: int
    y: int

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return [self.x, self.y][index]

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    point1: Point
    point2: Point


class ObjectDetector:
    def __init__(self, yolo_version="yolov8n.pt", device="cuda") -> None:
        self.model = YOLO("./yolov8n.pt")
        self.device = device

    def __box_to_detected_object(self, box: Boxes) -> DetectedObject:
        return DetectedObject(
            class_name=self.model.names[int(box.cls)],
            confidence=int(box.conf),
            point1=Point(*map(int, box.xyxy[0, :2])),
            point2=Point(*map(int, box.xyxy[0, 2:])),
        )

    def __draw_bounding_boxes_in_frame(
        self, frame: np.ndarray, detection: Results
    ) -> None:
        for box in detection.boxes:
            detected_object = self.__box_to_detected_object(box)

            cv2.rectangle(
                frame,
                detected_object.point1,
                detected_object.point2,
                color=(255, 0, 0),
                thickness=2,
            )

            label = f"{detected_object.class_name}: {detected_object.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                org=(detected_object.point1.x, detected_object.point1.y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=2,
            )

    def show_video_with_bounding_boxes(self, video_path: str) -> None:
        video = cv2.VideoCapture(video_path)

        while True:
            video_status, frame = video.read()

            if not video_status:
                break

            detections = self.model(frame, stream=True, device=self.device)

            for detection in detections:
                self.__draw_bounding_boxes_in_frame(frame, detection)

            cv2.imshow("Frame with Bounding Boxes", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        video.release()
        cv2.destroyAllWindows()


# Below is a simple example to detect objects in a video.
# Use "mps" as a device for MacBooks and "cuda" for Nvidia GPUs.
# Hold 'q' to exit
if __name__ == "__main__":
    detector = ObjectDetector(device="mps")
    detector.show_video_with_bounding_boxes("./videos/football.mp4")

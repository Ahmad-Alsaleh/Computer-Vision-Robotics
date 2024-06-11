from dataclasses import dataclass
from typing import List, Optional
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import cv2
from rich import print
from similarity_finder import SimilarityFinder


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
    embedded_class_name: torch.Tensor


class ObjectDetector:
    def __init__(self, yolo_version="yolov8n.pt", device="cuda") -> None:
        self.model = YOLO(yolo_version)
        self.device = device
        self.similarity_finder = SimilarityFinder(device=device)

    def __box_to_detected_object(self, box: Boxes) -> DetectedObject:
        class_name = self.model.names[int(box.cls)]
        embedded_class_name = self.similarity_finder.embed_text(class_name)

        return DetectedObject(
            class_name=class_name,
            confidence=float(box.conf),
            point1=Point(*map(int, box.xyxy[0, :2])),
            point2=Point(*map(int, box.xyxy[0, 2:])),
            embedded_class_name=embedded_class_name,
        )

    def __draw_bounding_boxes_in_frame(
        self, frame: np.ndarray, results: Results
    ) -> None:
        for box in results.boxes:
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

    def __check_exit(self, key="q") -> None:
        return cv2.waitKey(1) & 0xFF == ord(key)

    def detect_objects(self, video_path: str, draw=False) -> List[List[DetectedObject]]:
        video = cv2.VideoCapture(video_path)

        # stores a list of detected objects for each frame
        bounding_boxes: List[List[DetectedObject]] = []

        while True:
            video_status, frame = video.read()

            if not video_status:
                break

            results: Results = next(self.model(frame, stream=True, device=self.device))

            bounding_boxes.append(
                [self.__box_to_detected_object(box) for box in results.boxes]
            )

            if draw:
                for result in results:
                    self.__draw_bounding_boxes_in_frame(frame, result)
                cv2.imshow("Frame with Bounding Boxes", frame)
                if self.__check_exit():
                    break

        # Release resources
        video.release()
        cv2.destroyAllWindows()

        return bounding_boxes


# Below is a simple example to detect objects in a video.
# Use "mps" as a device for MacBooks and "cuda" for Nvidia GPUs.
# Hold 'q' to exit
if __name__ == "__main__":
    detector = ObjectDetector(device="mps")
    bounding_boxes = detector.detect_objects("./videos/football 2.mp4", draw=True)
    print(bounding_boxes)

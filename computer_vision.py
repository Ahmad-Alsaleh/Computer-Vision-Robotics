import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2


class ObjectDetector:
    def __init__(self, yolo_version="yolov8n.pt", device="cuda") -> None:
        self.model = YOLO("./yolov8n.pt")
        self.device = device

    def __draw_bounding_boxes_in_frame(
        self, frame: np.ndarray, detection: Results
    ) -> None:
        for box in detection.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = int(box.conf)

            class_id = int(box.cls)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                color=(255, 0, 0),
                thickness=2,
            )

            label = f"{self.model.names[class_id]}: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                org=(x_min, y_min - 5),
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


if __name__ == "__main__":
    # Below is a simple example to detect objects in a video.
    # Use "mps" as a device for MacBooks and "cuda" for Nvidia GPUs.
    detector = ObjectDetector(device="mps")
    detector.show_video_with_bounding_boxes("./videos/football.mp4")

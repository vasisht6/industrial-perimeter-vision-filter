import cv2
import json
import os
import sys
from src.perimeter_filter import PerimeterVisionFilter
from src.roi_selector import select_roi

def main():
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    config_file = "roi_config.json"

    if not os.path.exists(config_file):
        print("[INFO] No ROI configuration found. Launching selector...")
        select_roi(video_source, config_file)

    with open(config_file, "r") as f:
        roi = json.load(f)["roi"]

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {video_source}")
        return

    engine = PerimeterVisionFilter(roi_polygon=roi, min_area=1500, persistence_thresh=4, history_len=6)

    print("[INFO] Processing stream. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_breach, annotated_frame = engine.process(frame)
        cv2.imshow("Industrial Perimeter Gate", annotated_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
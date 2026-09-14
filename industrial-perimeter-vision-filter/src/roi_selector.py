import cv2
import json
import numpy as np

pts = []

def click_event(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(param, (x, y), 4, (0, 255, 0), -1)
        if len(pts) > 1:
            cv2.line(param, tuple(pts[-2]), tuple(pts[-1]), (255, 0, 0), 2)
        cv2.imshow("Select ROI - Press 'q' to save, 'r' to reset", param)

def select_roi(video_path: str, output_json: str = "roi_config.json"):
    global pts
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not load frame from {video_path}")

    clone = frame.copy()
    cv2.namedWindow("Select ROI - Press 'q' to save, 'r' to reset")
    cv2.setMouseCallback("Select ROI - Press 'q' to save, 'r' to reset", click_event, clone)

    while True:
        cv2.imshow("Select ROI - Press 'q' to save, 'r' to reset", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clone = frame.copy()
            pts = []
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(pts) < 3:
        raise ValueError("A polygon needs at least 3 points.")

    with open(output_json, "w") as f:
        json.dump({"roi": pts}, f, indent=4)
    print(f"[INFO] Saved {len(pts)} boundary points to {output_json}")

if __name__ == "__main__":
    import sys
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    select_roi(video_source)
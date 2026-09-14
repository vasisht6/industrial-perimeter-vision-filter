import cv2
import numpy as np
from collections import deque

class PerimeterVisionFilter:
    def __init__(self, roi_polygon: list, min_area: int = 1500, persistence_thresh: int = 4, history_len: int = 6):
        """
        roi_polygon: List of [x, y] coordinates forming the perimeter boundary.
        min_area: Minimum contour pixel area to count as an intruder (rejects rain/insects).
        persistence_thresh: Required breach hits inside the temporal window (K).
        history_len: Total rolling window length (N).
        """
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32)
        self.min_area = min_area
        self.persistence_thresh = persistence_thresh
        self.history = deque(maxlen=history_len)

        # MOG2 isolates background vs. foreground using a Mixture of Gaussians
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=False
        )
        # Rectangular kernel for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.mask = None

    def _init_mask(self, frame_shape):
        """Pre-allocates an 8-bit binary mask of the ROI polygon."""
        self.mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.roi_polygon], 255)

    def process(self, frame: np.ndarray):
        if self.mask is None:
            self._init_mask(frame.shape)

        # 1. Spatial Gating: Mask out all areas outside the perimeter polygon
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)

        # 2. Preprocessing: Blur reduces high-frequency sensor noise
        blurred = cv2.GaussianBlur(masked_frame, (5, 5), 0)

        # 3. Background Subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # 4. Morphological Cleaning: Remove speckles (Opening) then consolidate targets (Dilation)
        fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_cleaned = cv2.dilate(fg_cleaned, self.kernel, iterations=2)

        # 5. Contour Detection & Spatial Area Thresholding
        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_breached = False
        active_boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                frame_breached = True
                x, y, w, h = cv2.boundingRect(cnt)
                active_boxes.append((x, y, w, h))

        # 6. Temporal Persistence (K-of-N sliding window)
        self.history.append(1 if frame_breached else 0)
        current_hits = sum(self.history)
        sustained_breach = current_hits >= self.persistence_thresh

        # 7. Annotation & HUD Overlay
        output_frame = frame.copy()
        poly_color = (0, 0, 255) if sustained_breach else (0, 255, 0)
        cv2.polylines(output_frame, [self.roi_polygon], isClosed=True, color=poly_color, thickness=2)

        for (x, y, w, h) in active_boxes:
            box_color = (0, 0, 255) if sustained_breach else (0, 255, 255)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), box_color, 2)

        status_text = f"STATE: BREACH ({current_hits}/{self.persistence_thresh})" if sustained_breach else "STATE: SECURE"
        cv2.putText(output_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, poly_color, 2)

        return sustained_breach, output_frame
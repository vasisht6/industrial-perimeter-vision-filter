# Deterministic Perimeter Vision Filter (Industrial Surveillance Gating)

An edge-optimized computer vision pre-filter designed for perimeter intrusion detection. Implements polygon spatial gating, morphological artifact suppression, and sliding-window temporal persistence ($K$-of-$N$ buffer) to eliminate false alarms from transient optical noise prior to downstream compute triggers.

## Key Architecture

1. **Polygon Spatial ROI Gating:** `cv2.bitwise_and` zero-copy binary masking eliminates off-premise motion before background subtraction.
2. **Foreground Extraction:** Dynamic Mixture of Gaussians (MOG2) background model with adaptive shadow rejection.
3. **Morphological Filtering:** Morphological Opening ($5 \times 5$ kernel) eliminates isolated sensor noise; Dilation consolidates disjointed human silhouettes.
4. **Spatial Area Thresholding:** Area contours $< 1500 \text{ px}$ are rejected to suppress rain streaks and lens insects.
5. **Temporal Persistence Engine:** Sliding FIFO buffer ($N=6$, $K=4$) requires 4 positive breach detections within 6 consecutive frames to raise an alert, suppressing transient single-frame anomalies.

## Benchmark & Performance

- **Throughput:** ~115 FPS @ 1080p on single-threaded x86 CPU.
- **Memory Footprint:** < 45 MB RAM usage.
- **Hardware Requirement:** Zero GPU dependency; deployable on edge IPCs and ARM gateways.

## Getting Started

```bash
pip install -r requirements.txt
python main.py path/to/video.mp4

==========================================================
Nuts and Bolts Dataset (Version 2)
==========================================================

Dataset Name : NutsAndBolts
Version      : 2
Export Date  : October 29, 2023
Format        : YOLO v5 PyTorch

Source:
https://universe.roboflow.com/kletech-university/nutsandbolts

License:
CC BY 4.0

==========================================================
Dataset Information
==========================================================

Total Images : 3245

Classes:
  • bolt
  • nut

Annotation Format:
YOLO v5 PyTorch

==========================================================
Pre-processing
==========================================================

• Auto Orientation (EXIF Removed)
• Resize Images to 416 × 416

==========================================================
Data Augmentation
==========================================================

• Horizontal Flip (50%)
• Vertical Flip (50%)
• 90° Rotation (Random)
• Brightness Adjustment (-45% to +45%)
• Exposure Adjustment (-25% to +25%)
• Gaussian Blur (0–1 px)
• Salt & Pepper Noise (2%)

==========================================================
Project Purpose
==========================================================

This dataset is used to train a YOLO object detection model
for detecting two object classes:

1. bolt
2. nut

The trained model is integrated with a FastAPI backend and
a web frontend for real-time bolt and nut detection.

==========================================================
Generated Using
==========================================================

Roboflow Universe
https://universe.roboflow.com

Training Framework:
Ultralytics YOLO

==========================================================

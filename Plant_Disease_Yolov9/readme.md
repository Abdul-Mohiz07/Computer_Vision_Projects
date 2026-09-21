# 🌱 Plant Disease Detection using YOLOv9s

A computer vision project for **plant leaf and disease detection using YOLOv9s**. The model detects the target class, identifies its location using a bounding box, and provides a confidence score for each prediction.

This project uses the **PlantDoc dataset** and trains a 30-class YOLOv9s object detection model in Google Colab using a Tesla T4 GPU.

---

## 📌 Project Overview

Traditional image classification answers:

> **"What is in this image?"**

This project goes one step further and performs **object detection**, answering:

> **"What is it, where is it, and how confident is the model?"**

### Example Output

```text
Class: Tomato Early blight leaf
Confidence: 87%

Bounding Box:
[x1, y1, x2, y2]
```

The final prediction is visualized directly on the input image with a bounding box and class label.

---

## 🎯 Objectives

* Detect plant leaves and disease-related classes.
* Locate detected objects using bounding boxes.
* Predict the corresponding class.
* Generate confidence scores for detections.
* Train and evaluate a YOLOv9s object detection model.
* Understand the complete object detection workflow from annotation to inference.

---

## 🗂️ Dataset

### PlantDoc

The project uses the **PlantDoc dataset**, which contains images of plant leaves and disease categories annotated with bounding boxes.

**Dataset format:** YOLO

**Number of classes:** 30

**Image resolution:** 416 × 416

### Classes

| ID | Class                                |
| -: | ------------------------------------ |
|  0 | Apple Scab Leaf                      |
|  1 | Apple leaf                           |
|  2 | Apple rust leaf                      |
|  3 | Bell_pepper leaf                     |
|  4 | Bell_pepper leaf spot                |
|  5 | Blueberry leaf                       |
|  6 | Cherry leaf                          |
|  7 | Corn Gray leaf spot                  |
|  8 | Corn leaf blight                     |
|  9 | Corn rust leaf                       |
| 10 | Peach leaf                           |
| 11 | Potato leaf                          |
| 12 | Potato leaf early blight             |
| 13 | Potato leaf late blight              |
| 14 | Raspberry leaf                       |
| 15 | Soyabean leaf                        |
| 16 | Soybean leaf                         |
| 17 | Squash Powdery mildew leaf           |
| 18 | Strawberry leaf                      |
| 19 | Tomato Early blight leaf             |
| 20 | Tomato Septoria leaf spot            |
| 21 | Tomato leaf                          |
| 22 | Tomato leaf bacterial spot           |
| 23 | Tomato leaf late blight              |
| 24 | Tomato leaf mosaic virus             |
| 25 | Tomato leaf yellow virus             |
| 26 | Tomato mold leaf                     |
| 27 | Tomato two spotted spider mites leaf |
| 28 | grape leaf                           |
| 29 | grape leaf black rot                 |

---

## 🧠 Model

### YOLOv9s

The project uses **YOLOv9s**, a lightweight YOLO model suitable for real-time object detection applications.

### Training Configuration

| Parameter         | Value           |
| ----------------- | --------------- |
| Model             | YOLOv9s         |
| Epochs            | 50              |
| Image Size        | 416 × 416       |
| Batch Size        | 16              |
| GPU               | NVIDIA Tesla T4 |
| Number of Classes | 30              |
| Framework         | Ultralytics     |
| Dataset Format    | YOLO            |

---

## 🔄 Object Detection Pipeline

```text
PlantDoc Dataset
       │
       ▼
YOLO Bounding Box Annotations
       │
       ▼
Train / Validation / Test Split
       │
       ▼
YOLOv9s
       │
       ▼
Feature Extraction
       │
       ▼
Object Detection
       │
       ├── Class
       ├── Bounding Box
       └── Confidence
       │
       ▼
Final Detection
```

---

## 📦 YOLO Annotation Format

Each object is represented using:

```text
class_id center_x center_y width height
```

Example:

```text
23 0.35625 0.56783 0.7025 0.71705
```

Where:

* `23` → class ID
* `0.35625` → bounding box center X
* `0.56783` → bounding box center Y
* `0.7025` → bounding box width
* `0.71705` → bounding box height

The coordinates are normalized between **0 and 1**.

---

# 📊 Results

The trained YOLOv9s model was evaluated on the held-out **test set**.

### Overall Test Performance

| Metric         |     Score |
| -------------- | --------: |
| Precision      | **62.5%** |
| Recall         | **58.1%** |
| mAP@50         | **61.9%** |
| mAP@50:95      | **48.9%** |
| Test Images    |   **239** |
| Test Instances |   **454** |

### What the Metrics Mean

**Precision**

Measures how many of the model's detections are correct.

**Recall**

Measures how many of the actual objects the model successfully detects.

**mAP@50**

Mean Average Precision calculated at an IoU threshold of 0.50.

**mAP@50:95**

Mean Average Precision calculated across IoU thresholds from 0.50 to 0.95, providing a stricter evaluation of both detection and localization quality.

---

## 🏆 Selected Class-Level Results

Some classes achieved particularly strong test-set detection performance:

| Class                | Precision | Recall | mAP@50 | mAP@50:95 |
| -------------------- | --------: | -----: | -----: | --------: |
| Strawberry leaf      |     98.9% |   100% |  99.5% |     84.8% |
| Corn rust leaf       |     99.2% |  90.0% |  98.6% |     83.5% |
| Grape leaf black rot |     85.1% |  71.6% |  90.9% |     76.1% |
| Peach leaf           |     61.9% |  90.0% |  89.9% |     69.1% |
| Corn leaf blight     |     69.6% |  83.3% |  87.7% |     77.4% |

Class-level performance varies because the dataset contains different numbers of examples and visually similar categories.

---

# 🖼️ Detection Output

The model produces:

```text
Input Image
     │
     ▼
YOLOv9s
     │
     ▼
┌─────────────────────────┐
│ Bounding Box            │
│ Class                   │
│ Confidence Score        │
└─────────────────────────┘
```

Example:

```text
Tomato Early blight leaf
Confidence: 0.87
```

> Add your generated prediction image here after running inference.

```markdown
![YOLO Prediction](examples/sample_prediction.jpg)
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Plant-Disease-Detection-YOLOv9.git
cd Plant-Disease-Detection-YOLOv9
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies

```text
ultralytics
torch
torchvision
opencv-python
matplotlib
```

---

# 🚀 Training

The model was trained using Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("yolov9s.pt")

results = model.train(
    data="/content/PlantDoc/data.yaml",
    epochs=50,
    imgsz=416,
    batch=16,
    device=0
)
```

The best-performing weights are saved as:

```text
runs/plantdoc_yolov9s/weights/best.pt
```

---

# 🔍 Inference

Load the trained model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

Run detection:

```python
results = model.predict(
    source="leaf.jpg",
    conf=0.25,
    imgsz=416,
    device=0,
    save=True
)
```

The model returns:

* Detected class
* Bounding box coordinates
* Confidence score

---

# 📁 Project Structure

```text
Plant-Disease-Detection-YOLOv9/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Plant_Disease_Detection_YOLOv9.ipynb
│
├── inference/
│   └── predict.py
│
├── models/
│   └── best.pt
│
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── predictions/
│
└── examples/
    └── sample_prediction.jpg
```

---

# 🛠️ Technologies Used

* **Python**
* **YOLOv9s**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **Google Colab**
* **CUDA**
* **NVIDIA Tesla T4**
* **PlantDoc Dataset**

---

# 📚 Key Computer Vision Concepts Applied

This project provided practical experience with:

* Object detection
* Bounding box annotations
* YOLO label format
* Train/validation/test splitting
* CNN-based feature extraction
* Confidence scores
* Intersection over Union (IoU)
* Non-Maximum Suppression (NMS)
* Precision
* Recall
* Average Precision (AP)
* mAP@50
* mAP@50:95
* Model validation
* Model testing
* Image inference

---

# ⚠️ Limitations

The model's performance varies considerably across the 30 classes.

Some classes have relatively few test examples, while others have substantially more. Visually similar plant and disease categories can also be challenging to distinguish.

The reported metrics should therefore be interpreted as performance on this specific PlantDoc test split rather than as a guarantee of performance on new agricultural environments.

The model is intended as a **computer vision project and detection system**, not as a definitive agricultural or plant-health diagnosis tool.

---

# 🔮 Future Improvements

Potential improvements include:

* Increase training data for underrepresented classes.
* Improve class balance.
* Experiment with YOLOv9m or other model variants.
* Tune confidence and IoU thresholds.
* Apply additional data augmentation.
* Perform hyperparameter tuning.
* Analyze confusion between visually similar classes.
* Test on real-world images captured outside the dataset.
* Build a web-based inference interface.
* Deploy the model for real-time camera/video detection.
* Optimize the model for edge/mobile deployment.

---

# 👨‍💻 Author

**Abdul Mohiz**

BS Data Science Student | Machine Learning & Computer Vision

Interested in:

* Data Science
* Machine Learning
* Deep Learning
* Computer Vision
* AI Applications

---

## ⭐ Project Summary

This project demonstrates a complete **object detection workflow for plant leaf and disease-related classes**, from YOLO annotation and dataset preparation to model training, validation, testing, and real-time-style image inference.

### Final Test Result

```text
YOLOv9s
30 Classes
50 Epochs
416 × 416 Images

Precision:    62.5%
Recall:       58.1%
mAP@50:       61.9%
mAP@50:95:    48.9%
```

---


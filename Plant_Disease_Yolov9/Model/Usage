# YOLOv9s Trained Model

This folder contains the trained **YOLOv9s model** fine-tuned on the PlantDoc dataset for 30-class plant leaf and disease-related object detection.

## Model

**File:** `best.pt`

The model was trained for:

* **Model:** YOLOv9s
* **Dataset:** PlantDoc
* **Classes:** 30
* **Epochs:** 50
* **Image Size:** 416 × 416
* **Framework:** Ultralytics YOLO
* **GPU:** NVIDIA Tesla T4

## How to Use

Install Ultralytics:

```bash
pip install ultralytics
```

Load the trained model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

Run detection on an image:

```python
results = model.predict(
    source="leaf.jpg",
    conf=0.25,
    imgsz=416
)
```

The model returns:

* Predicted class
* Bounding box
* Confidence score

You can also display the prediction:

```python
import matplotlib.pyplot as plt

result_image = results[0].plot()

plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis("off")
plt.show()
```

The trained weights can be used directly for inference without retraining the model.


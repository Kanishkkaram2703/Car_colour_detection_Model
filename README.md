# Car Color Detection

## 1. Abstract

This project presents a machine learning solution for real-time car color detection and people counting at traffic signals. The system classifies car colors and highlights them with dynamic annotations: red rectangles for blue cars and blue rectangles for others. It also detects and counts people using Haar cascade classifiers. A user-friendly GUI is developed using Tkinter to provide visual feedback of the detection.

## 2. Introduction

### Problem Statement:
Accurate vehicle and pedestrian recognition is essential for intelligent traffic monitoring. Identifying car colors and counting traffic participants (vehicles and people) can support traffic analysis, automation, and safety systems.

### Objectives:

- Train a CNN model for car color classification.
- Detect cars and pedestrians using Haar cascades.
- Annotate vehicles based on color.
- Create a GUI interface to upload images and preview output.

### Applications:

- Smart traffic signals
- Real-time vehicle classification systems
- Urban traffic data analysis

## 3. Literature Review

- **Car Color Classification with CNNs**: Deep learning models like CNNs provide high accuracy in image-based color classification.
- **Haar Cascades for Object Detection**: Lightweight and fast for vehicle and pedestrian detection.
- **GUI in Traffic Systems**: Interfaces enhance usability and interpretation of results.

## 4. Methodology

### Dataset:
- The car color classification dataset is sourced from Kaggle with labeled car images in various colors.
- Classes: ['black', 'blue', 'brown', 'gold','green', 'grey', 'orange', 'pink','purple', 'red', 'silver', 'tan','white', 'yellow', 'beige']

### Preprocessing:
- Resizing images to 224x224
- Normalizing pixel values to [0,1]
- Augmentations: flipping, rotation, contrast changes

### Model Architecture:
- CNN with layers: Conv2D, MaxPooling, Flatten, Dense
- Loss: Categorical Crossentropy
- Optimizer: Adam

### Detection Logic:
- Load the trained model
- Use Haar cascades (`cars.xml`, `haarcascade_fullbody.xml`) to detect cars and people
- Classify color of detected car regions
- Annotate frames:
  - 🔵 Blue Rectangle → Non-blue cars
  - 🔴 Red Rectangle → Blue cars
  - 🟢 Green Rectangle → People

### GUI:
- Built with Tkinter
- Upload images
- Preview processed image with annotations

## 5. Implementation

### Core Files:

- `gui.py`: Tkinter-based app for GUI + detection logic
- `model_training.ipynb`: Jupyter Notebook used for training car color classifier
- `cars.xml` and `haarcascade_fullbody.xml`: Haar cascades for detection

### Sample Classification Code:

```python
def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)
```
**6. Results**
Training Accuracy: ~92%

Validation Accuracy: ~88%

Real-time car and people detection in images

Correct annotations using colored rectangles

**7. Challenges and Limitations**
Challenges:
Lighting variance in traffic images

Low resolution or blurred car images

Similar color shades causing misclassification

Limitations:
Haar cascades might miss objects in complex backgrounds

No live webcam or video stream by default (image-based only)

**8. Conclusion**
This system demonstrates successful integration of a CNN-based classifier and traditional object detectors to build a smart traffic detection app. The GUI interface makes it user-friendly and visually informative.

🔄 Pre-trained Model
You can download the trained .keras model from Google Drive:
📦 Google Drive - Car Color Detection Model

🚀 Features
Car color classification using a CNN model

Dynamic bounding box annotations

Detection of people at traffic signals

GUI for image upload and output display

📁 Dataset
Kaggle Vehicle Color Recognition dataset

Download and place it under dataset/ directory

✅ Requirements
Python 3.8+

OpenCV

TensorFlow

NumPy

Pillow

Tkinter (built-in)

Matplotlib (for training analysis)

💾 Installation

git clone https://github.com/Kanishkkaram2703/Car-color-detection.git
cd Car-color-detection
pip install -r requirements.txt
🧠 Model Training
To train the car color classifier:

Download the dataset from Kaggle

Place it under the dataset/ folder

Open model_training.ipynb and run all cells

Save the final model as .keras

Or download the pretrained model:
📥 Google Drive - Trained Model
## 🔄 Pre-trained Model

Download the trained `.keras` model directly from this Google Drive folder:  
📦 [Google Drive - Car Color Detection](https://drive.google.com/drive/folders/1qsjD8CMuT5IU3eQ2XtIxaIE2TESI6iet?usp=drive_link)


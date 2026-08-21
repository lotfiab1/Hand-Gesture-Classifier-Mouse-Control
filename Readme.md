# Hand Gesture Mouse Control ✋🖱️

A real-time **computer vision and machine learning** application that allows you to control your computer mouse using hand gestures and a webcam — without a physical mouse.

The system uses **MediaPipe** to detect hand landmarks, **scikit-learn** to classify gestures, and **PyAutoGUI** to translate predictions into mouse actions.

## 🎥 Demo

The application tracks your hand in real time and converts recognized gestures into mouse interactions.

**Supported interactions:**

| Gesture       | Action                    |
| ------------- | ------------------------- |
| `click`       | Left mouse click          |
| `scroll_up`   | Scroll up                 |
| `scroll_down` | Scroll down               |
| `release`     | Reserved for drag/release |

> **Note:** The `release` gesture is currently reserved for future drag-and-drop functionality.

## ⚙️ How It Works

The application follows this pipeline:

```text
Webcam
   ↓
MediaPipe HandLandmarker
   ↓
21 Hand Landmarks
   ↓
Landmark Normalization
   ↓
Machine Learning Classifier
   ↓
Gesture Prediction
   ↓
Mouse Action
```

### 1. Hand Tracking

`HandLandMarks.py` uses **MediaPipe HandLandmarker** to detect the user's hand from the webcam and extract **21 three-dimensional landmarks** for each frame.

The implementation supports:

* Live-stream mode
* Video mode
* Single-image mode

### 2. Landmark Normalization

`landmark_utils.py` preprocesses the detected landmarks before classification.

The landmarks are:

* Centered relative to the wrist
* Scaled to reduce the effect of hand size and camera distance
* Converted into features suitable for machine learning

This helps the classifier focus on the **shape of the gesture** rather than its position in the camera frame.

### 3. Gesture Classification

A **scikit-learn** machine learning pipeline is used to classify the normalized landmarks.

The project evaluates:

* Logistic Regression
* Random Forest

The best-performing model is selected through cross-validation in `retrain_model.py`.

The classifier predicts one of the supported gestures:

```text
click
scroll_up
scroll_down
release
```

### 4. Mouse Control

`main.py` connects the gesture predictions to the operating system using **PyAutoGUI**.

The hand's bounding-box center is mapped to the screen position, allowing the user to control the cursor naturally.

An **exponential moving average** is used to smooth cursor movement and reduce jitter.

## 🧠 Machine Learning Pipeline

```text
Hand Landmarks
      ↓
Feature Extraction
      ↓
Normalization
      ↓
Train / Validation
      ↓
Cross-Validation
      ↓
Best Model Selection
      ↓
Real-Time Prediction
```

This approach allows the system to learn gesture patterns instead of relying entirely on manually defined rules.

## 🛠️ Technologies

* **Python**
* **OpenCV** — webcam and image processing
* **MediaPipe** — hand landmark detection
* **scikit-learn** — gesture classification
* **PyAutoGUI** — mouse automation
* **NumPy** — numerical processing

## 📁 Project Structure

```text
Hand-Gesture-Classifier-Mouse-Control/
│
├── main.py
├── HandLandMarks.py
├── landmark_utils.py
├── retrain_model.py
├── dataset.py
│
├── models/
│   └── ...
│
├── data/
│   └── ...
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/lotfiab1/Hand-Gesture-Classifier-Mouse-Control.git

cd Hand-Gesture-Classifier-Mouse-Control
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

Make sure your webcam is connected and accessible.

## 🎮 Usage

1. Start the application.
2. Allow access to your webcam.
3. Place your hand in front of the camera.
4. Move your hand to control the cursor.
5. Perform a supported gesture to trigger a mouse action.

For the best experience:

* Use good lighting.
* Keep your hand clearly visible.
* Avoid excessive background movement.
* Keep a reasonable distance from the camera.

## 🔬 Training the Model

The project includes tools for retraining the gesture classifier.

The general workflow is:

```text
Collect Gesture Data
        ↓
Extract Hand Landmarks
        ↓
Normalize Features
        ↓
Train Classifiers
        ↓
Cross-Validation
        ↓
Select Best Model
```

To retrain the model:

```bash
python retrain_model.py
```

You can collect additional samples and retrain the classifier to improve recognition for different users, environments, or hand positions.

## 🔮 Future Improvements

Planned improvements include:

* [ ] Right-click gesture
* [ ] Double-click gesture
* [ ] Drag-and-drop support
* [ ] Gesture-based application shortcuts
* [ ] More robust gesture recognition
* [ ] Improved cursor calibration
* [ ] Multi-hand interaction
* [ ] Real-time model performance metrics
* [ ] Improved UI for configuration and calibration

## 📌 Project Goals

This project was developed to explore the combination of:

* Real-time computer vision
* Hand tracking
* Machine learning classification
* Feature engineering
* Human-computer interaction
* Desktop automation

The goal is to demonstrate how machine learning and computer vision can be used to create practical, interactive applications.

## 👨‍💻 Author

**Lotfi Ait Baaya**

Junior Data Analyst & AI Developer

* GitHub: https://github.com/lotfiab1
* Portfolio: https://lotfiab1.github.io/lotfi-protfolio/

---

Disclaimer: This project is open source and available for educational and personal use. [LICENSE](./blob/master/LICENSE)

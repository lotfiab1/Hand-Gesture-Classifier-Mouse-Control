# Contrôle Mousse With Hand Gesture ✋🖱️

Control your mouse cursor and clicks using nothing but your webcam and hand gestures. The project tracks your hand in real time with **MediaPipe**, classifies the gesture with a **scikit-learn** model, and drives the system cursor with **PyAutoGUI**.

## How it works

1. **Hand tracking** — `HandLandMarks.py` wraps MediaPipe's `HandLandmarker` (live-stream, video, and single-image modes) to extract 21 3D hand landmarks per frame from the webcam feed.
2. **Normalization** — `landmark_utils.py` centers the landmarks on the wrist and scales them so the classifier is invariant to hand position and distance from the camera.
3. **Classification** — a scikit-learn pipeline (best of Logistic Regression / Random Forest, chosen by cross-validation in `retrain_model.py`) predicts one of four gestures from the normalized landmarks.
4. **Control** — `main.py` moves the cursor to the (smoothed) center of the hand's bounding box every frame, and triggers mouse actions when a gesture is predicted with high confidence.

### Supported gestures

| Gesture label | Action |
|---|---|
| `click` | Left mouse click |
| `release` | Reserved for drag/release (currently disabled in code) |
| `scroll_up` | Scroll up |
| `scroll_down` | Scroll down |

Cursor movement is continuous and always active — it follows the bounding box of the detected hand, smoothed with an exponential moving average to avoid jitter.

## Demo

A short walkthrough video is included:

<video src="./blob/master/video_tutorial.mp4" controls width="100%"></video>

Sample captures used for building the dataset are in [`image sample/`](image%20sample).

## Project structure

```
Controle_Mousse_With_Hand_Gesture/
├── main.py                # Real-time gesture-controlled mouse (entry point)
├── dataset.py              # Webcam-based dataset collection utility
├── retrain_model.py         # Trains/evaluates the classifier and saves the model
├── landmark_utils.py        # Landmark normalization helpers
├── HandLandMarks.py         # MediaPipe HandLandmarker wrapper (image/video/live-stream)
├── notebook.ipynb           # Exploratory training/analysis notebook
├── requirements.txt
├── Dataset/
│   ├── data.csv              # Training data
│   ├── validation.csv         # Validation data
│   ├── samples.csv / samples2.csv  # Raw collected samples
├── resources/
│   ├── hand_landmarker.task    # MediaPipe hand landmark detection model
│   └── model.pl                # Trained gesture classifier (joblib)
├── image sample/             # Example frames from data collection
└── video_tutorial.mp4        # Demo/tutorial video
```

## Installation

Requires **Python 3.9+**.

```bash
git clone <this-repo-url>
cd Controle_Mousse_With_Hand_Gesture
pip install -r requirements.txt
```

### Dependencies

- `mediapipe` — hand landmark detection
- `opencv-python` — webcam capture and drawing
- `pyautogui` — cursor and mouse control
- `pandas`, `numpy` — data handling
- `scikit-learn`, `joblib` — model training and persistence
- `matplotlib` — used in the exploratory notebook

## Usage

### Run gesture mouse control

```bash
python main.py
```

- A camera window opens showing the live feed with hand landmarks, bounding box, and the predicted gesture + confidence.
- Move your hand to move the cursor; hold a recognized gesture to trigger its action.
- Press `q` or close the window to quit.

> ⚠️ **Safety note:** `pyautogui.FAILSAFE` is enabled — move the mouse to a screen corner at any time to abort if the cursor misbehaves.

### Collect your own training data

```python
from HandLandMarks import HandLandmarksHandleLiveStream
from dataset import generate_dataset

labels = ["click", "release", "scroll_up", "scroll_down"]

with HandLandmarksHandleLiveStream() as landmarker:
    generate_dataset(landmarker, labels, "Dataset/data.csv", num_samples=200)
```

This opens the webcam and records normalized landmark samples for each label in turn (with a countdown between labels), saving them to CSV.

### Retrain the model

```bash
python retrain_model.py
```

This loads `Dataset/data.csv` and `Dataset/validation.csv`, cross-validates Logistic Regression and Random Forest pipelines, picks the best performer on the validation set, prints per-label accuracy, and saves the winning model to `resources/model.pl`.

## Configuration

Key tunables live at the top of `main.py`:

| Variable | Purpose | Default |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | Minimum model confidence to act on a prediction | `0.75` |
| `SMOOTHING` | Cursor movement smoothing factor (0–1, higher = snappier) | `0.35` |
| `EDGE_MARGIN` | Pixel margin keeping the cursor off screen edges | `2` |

## Notes & limitations

- Currently tuned for a **single hand**; only the first detected hand drives the cursor.
- The `release`/drag-hold logic is present in the code but commented out — clicks are simple press events, not click-and-hold.
- Performance depends on lighting and camera quality, since it relies entirely on MediaPipe's hand landmark detection.

## License

This project is open source and available for educational and personal use. [LICENSE](./blob/master/LICENSE)

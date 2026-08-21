from HandLandMarks import HandLandmarksHandleLiveStream
from landmark_utils import normalize_landmarks
import numpy as np
import joblib
import time
import cv2
import pyautogui


labels = {0: "click", 1: "release", 2: "scroll_up", 3: "scroll_down"}

model_path = 'resources/model.pl'


CONFIDENCE_THRESHOLD = 0.75

SMOOTHING = 0.35

EDGE_MARGIN = 2

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class CursorState:

    def __init__(self):
        self.smoothed_x = None
        self.smoothed_y = None
        self.is_dragging = False
        self.last_label = None


def move_mousse(state, w_img, h_img, p1, p2):

    w, h = pyautogui.size()

    center_x = p1[0] + (p2[0] - p1[0]) / 2
    center_y = p1[1] + (p2[1] - p1[1]) / 2

    target_x = np.clip((center_x / w_img) * w, EDGE_MARGIN, w - EDGE_MARGIN)
    target_y = np.clip((center_y / h_img) * h, EDGE_MARGIN, h - EDGE_MARGIN)

    if state.smoothed_x is None:
        state.smoothed_x, state.smoothed_y = target_x, target_y
    else:
        state.smoothed_x += (target_x - state.smoothed_x) * SMOOTHING
        state.smoothed_y += (target_y - state.smoothed_y) * SMOOTHING

    pyautogui.moveTo(state.smoothed_x, state.smoothed_y, duration=0)


def rectangle(img, p1, p2):
    cv2.rectangle(img, p1, p2, (255, 255, 255), 2)


def text(img, text_, p):
    cv2.putText(img, text_, p, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def shouldClose(wnd_name) -> bool:
    if cv2.waitKey(1) & 0xff == ord('q') or cv2.getWindowProperty(wnd_name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False


def handle_action(state, predicted_label):

    if predicted_label == state.last_label:
        if predicted_label == "scroll_up":
            pyautogui.scroll(50)
        elif predicted_label == "scroll_down":
            pyautogui.scroll(-50)
        return

    if predicted_label == "click" and not state.is_dragging:
        pyautogui.click()
        #state.is_dragging = True
    """
    elif predicted_label == "release" and state.is_dragging:
        pyautogui.mouseUp()
        state.is_dragging = False
    """
    state.last_label = predicted_label


def openWindow(cap, landmarker, model, labels, state):
    wnd_name = 'Camera Viewer'
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = landmarker.ToMpImage(rgb_frame)
        timestamps = int(time.time() * 1000)
        result = landmarker.getLandmarks(mp_img, timestamps)

        if result and result.hand_landmarks:

            for hand in result.hand_landmarks:

                landmarks = [[l.x, l.y, l.z] for l in hand]
                features = normalize_landmarks(landmarks).reshape(1, -1)

                proba = model.predict_proba(features)[0]
                pred_idx = int(np.argmax(proba))
                confidence = proba[pred_idx]

                p1, p2 = landmarker.getHandBoundingBox(w, h, hand)
                text_x = p1[0] + int((p2[0] - p1[0]) / 2)
                text_y = p1[1] - 10

                rectangle(frame, p1, p2)
                landmarker.drawLandmarks(frame, hand)

                move_mousse(state, w, h, p1, p2)

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = labels.get(pred_idx, "Not Found")
                    text(frame, f'{predicted_label} ({confidence:.2f})', (text_x, text_y))
                    handle_action(state, predicted_label)
                else:
                    text(frame, f'... ({confidence:.2f})', (text_x, text_y))

        cv2.imshow(wnd_name, frame)

        if shouldClose(wnd_name):
            break


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Failed Open Camera')
        exit(1)


    model = joblib.load(model_path)


    state = CursorState()

    with HandLandmarksHandleLiveStream() as landmarker:

        openWindow(cap, landmarker, model, labels, state)

    if state.is_dragging:
        pyautogui.mouseUp()

    cap.release()
    cv2.destroyAllWindows()

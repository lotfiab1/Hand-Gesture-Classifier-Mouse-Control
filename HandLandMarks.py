import ctypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarkerOptions,HandLandmarker


latest_result = None


class HandLandmarkerHandle:
    _instance = {}

    def __new__(cls, type_):

        if type_ not in cls._instance:
            cls._instance[type_] = super().__new__(cls)
        
        return cls._instance[type_]

    def __init__(self, options):
        if not hasattr(self,"Hands"):
            self.mp_hands = mp.tasks.vision.HandLandmarksConnections
            self.mp_drawing = mp.tasks.vision.drawing_utils
            self.mp_drawing_styles = mp.tasks.vision.drawing_styles
            self.Hands = HandLandmarker.create_from_options(options)
    def __enter__(self):
        return self
    def drawLandmarks(self,img,hand_landmarks,l_style = None,c_style = None):
        landmarks_style = self.mp_drawing_styles.get_default_hand_landmarks_style() if l_style is None else l_style
        connections_style = self.mp_drawing_styles.get_default_hand_connections_style() if c_style is None else c_style
        self.mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmarks_style,
            connections_style
        )
    def getHandBoundingBox(self,w,h,hand_landmarks,padding=28):
        x_coords = [landmark.x for landmark in hand_landmarks]
        y_coords = [landmark.y for landmark in hand_landmarks]

        p1 = (int(min(x_coords) * w) - padding,int(min(y_coords) * h) - padding)
        p2 = (int(max(x_coords) * w) + padding,int(max(y_coords) * h) + padding)

        return p1,p2
    def LoadImage(self,img_path):
        return mp.Image.create_from_file(img_path)
    def ToMpImage(self,img):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    def close(self):
        self.Hands.close()
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False



    
class HandLandmarksHandleImage(HandLandmarkerHandle):
    def __new__(cls):
        return super().__new__(cls,'Image')
    def __init__(self,model_path = 'resources/hand_landmarker.task'):
        options = HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands = 1,
                min_hand_detection_confidence= 0.5
            ) 
        super().__init__(options)
    def getLandmarks(self,img):
        result = self.Hands.detect(img)
        return result
class HandLandmarksHandleVideo(HandLandmarkerHandle):
    def __new__(cls):
        return super().__new__(cls,'Video')
    def __init__(self,model_path = 'resources/hand_landmarker.task'):
        options = HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands = 1
            ) 
        super().__init__(options)
    def getLandmarks(self,img,timestamp_ms):
            result = self.Hands.detect_for_video(img,timestamp_ms)
            return result
class HandLandmarksHandleLiveStream(HandLandmarkerHandle):
    def __new__(cls):
        return super().__new__(cls,'Live_Stream')
    def __init__(self,model_path = 'resources/hand_landmarker.task'):
        options = HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                result_callback = save_result_callback,
                num_hands = 1
            ) 
        super().__init__(options)
    def getLandmarks(self,img,timestamp_ms):
        self.Hands.detect_async(img,timestamp_ms)
        return latest_result


def save_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result



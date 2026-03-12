import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class CameraStream:
    def __init__(self):
        """
        Initialize webcam stream
        """
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def get_frame(self):
        """
        Capture a frame from the camera
        """
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        return frame

    def release(self):
        """
        Release camera resource
        """
        if self.cap:
            self.cap.release()

    def show_stream(self):
        """
        Simple camera preview (for testing)
        """
        while True:
            frame = self.get_frame()

            if frame is None:
                break

            cv2.imshow("Camera Stream", frame)

            key = cv2.waitKey(1) & 0xFF

            # Press Q to exit
            if key == ord('q'):
                break

        self.release()
        cv2.destroyAllWindows()


# Test camera directly
if __name__ == "__main__":
    camera = CameraStream()
    camera.show_stream()
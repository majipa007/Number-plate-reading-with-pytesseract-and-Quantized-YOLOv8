import cv2
import time
from detector import Detector


class Main:
    def __init__(self, path):
        self.path = path

    def run(self):
        print("hello")
        detector = Detector()
        cap = cv2.VideoCapture(self.path)
        p_time = 0
        if not cap.isOpened():
            print("Error: Unable to open the dash cam.")
            return

        while True:
            c_time = time.time()
            ret, frame = cap.read()
            frame = detector.pipeline(frame)

            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(frame, str(int(fps)), (10, 170), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255, 0, 255), 5)

            resized_frame = cv2.resize(frame, (1920, 1080))
            cv2.imshow('number-plate', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


x = Main("test.webm")
x.run()

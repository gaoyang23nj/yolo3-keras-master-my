import cv2
import matplotlib.pyplot as plt

def run_cap():
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_cap()

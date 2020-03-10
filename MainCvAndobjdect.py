import cv2
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

from dect import dect
from PIL import Image
from timeit import default_timer as timer

q_input=multiprocessing.Queue()
q_output=multiprocessing.Queue()

def run_cap_dec():
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        before = timer()

        ret, frame1 = cap.read()

        input_image = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        q_input.put(input_image)

        output_image = q_output.get(True)
        frame2 = cv2.cvtColor(np.asarray(output_image), cv2.COLOR_RGB2BGR)
        after = timer()
        # show a frame
        cv2.imshow("capture1", frame1)
        cv2.imshow("capture2", frame2)
        print(after-before)
        # cv2.imshow("capture2", result_element)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_fig_dec():
    # get a frame
    img_filepath = 'D:/13-ML/yolo3-keras-master-my/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    frame1 = cv2.imread(img_filepath)

    before = timer()

    input_image = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    q_input.put(input_image)

    output_image = q_output.get(True)
    frame2 = cv2.cvtColor(np.asarray(output_image), cv2.COLOR_RGB2BGR)
    end = timer()
    # show a frame
    cv2.imshow("capture1", frame1)
    cv2.imshow("capture2", frame2)
    # cv2.imshow("capture2", result_element)
    print(end-before)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    j = multiprocessing.Process(target=dect, args=(q_input, q_output))
    j.daemon = True
    j.start()

    cv2.waitKey(10)
    run_cap_dec()
    # test_fig_dec()



from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    # img = input('Input image filename:')
    img = 'D:\yolo3-keras-master/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
yolo.close_session()

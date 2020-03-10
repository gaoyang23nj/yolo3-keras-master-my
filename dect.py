import os
import numpy as np
import copy
import colorsys
import tensorflow as tf
import keras.backend as K

from nets.yolo3 import yolo_body, yolo_eval
from keras.layers import Input
from yolo import YOLO
from PIL import Image, ImageFont, ImageDraw
from keras.models import load_model
from timeit import default_timer as timer
from utils.utils import letterbox_image

global q_input
global q_output
# def dect(image, image_height, image_width):
def dect(q_input, q_output):
    #---------------------------------------------------#
    #   获得类和先验框
    #---------------------------------------------------#
    def get_classes(classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 获取classes和anchor的位置
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # 预训练模型的位置
    weights_path = 'logs_2stage_train/last1.h5'

    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # 输入的shape大小
    input_shape = (416, 416)

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    model_body.load_weights(weights_path, by_name=True)
    model_body.summary()

    print('{} model, anchors, and classes loaded.'.format(weights_path))


    # img = input('Input image filename:')
    # # img = 'D:\yolo3-keras-master/VOCdevkit/VOC2007/JPEGImages/000007.jpg'
    # image = Image.open(img)

    while True:
        em = q_input.get(True)
        if em is None:
            break
        image = em
        # 调整图片使其符合输入要求
        new_image_size = [416, 416]
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        # yolo_outputs = model_body(tf.convert_to_tensor(image_data))
        yolo_outputs = model_body.predict(image_data)

        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []

        from nets.yolo3 import yolo_boxes_and_scores
        from nets.yolo3 import yolo_head
        from nets.yolo3 import yolo_correct_boxes

        # width height
        image_shape = [image.size[1], image.size[0]]
        # image_shape = [image_height, image_width]

        for l in range(len(yolo_outputs)):
            feats = yolo_outputs[l]
            tmp_anchors = anchors[anchor_mask[l]]
            # num_classes
            # input_shape
            feats = tf.constant(feats)
            tmp_box_xy, tmp_box_wh, tmp_box_confidence, tmp_box_class_probs = yolo_head(feats, tmp_anchors, num_classes, input_shape)
            # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
            tmp_boxes = yolo_correct_boxes(tmp_box_xy, tmp_box_wh, input_shape, image_shape)
            tmp_boxes = K.reshape(tmp_boxes, [-1, 4])
            tmp_box_scores = tmp_box_confidence * tmp_box_class_probs
            tmp_box_scores = K.reshape(tmp_box_scores, [-1, num_classes])

            boxes.append(tmp_boxes)
            box_scores.append(tmp_box_scores)

        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)

        print(boxes.shape, box_scores.shape)

        score_threshold = .6
        max_boxes = 20
        iou_threshold =.5

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []

        for c in range(num_classes):
            # 取出所有box_scores >= score_threshold的框，和成绩
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

            # 非极大抑制，去掉box重合程度高的那一些
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

            # 获取非极大抑制后的结果
            # 下列三个分别是
            # 框的位置，得分与种类
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            # classes = K.ones_like(class_box_scores, 'int32') * c
            classes = tf.ones(class_box_scores.shape, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        out_boxes = K.concatenate(boxes_, axis=0)
        out_scores = K.concatenate(scores_, axis=0)
        out_classes = K.concatenate(classes_, axis=0)

        print('Found {} boxes for {}'.format(len(boxes_), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # font = ImageFont.truetype(font='font/simhei.ttf',
        #                           size=np.floor(3e-2 * image_height + 0.5).astype('int32'))
        # thickness = (image_width + image_height) // 300

        small_pic = []
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        num_object = out_boxes.shape[0]
        for i in range(num_object):
            box = out_boxes[i]
            score = out_scores[i]
            predicted_class = class_names[out_classes[i]]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # bottom = min(image_height, np.floor(bottom + 0.5).astype('int32'))
            # right = min(image_width, np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        # image.show()
        q_output.put(image)
    # return image




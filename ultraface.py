"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os
import subprocess
import time

import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils

# onnx runtime
import onnxruntime as ort


class UltraFace:

    def __init__(self, threshold=0.7, width=320, height=240):
        print("ULFG scheduling")
        show = f"chrt -p {os.getpid()}"
        os.system(show)

        label_path = "models/voc-model-labels.txt"
        self.width = width
        self.height = height
        self.onnx_path = f"models/onnx/version-RFB-{self.width}.onnx"
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.threshold = threshold

    def predict(self, width, height, confidences, boxes, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def detect(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, labels, probs = self.predict(img.shape[1], img.shape[0], confidences, boxes)
        return boxes

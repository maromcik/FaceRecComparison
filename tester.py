import os
import time
import cv2
import dlib
import random
from retinaface import RetinaFace
from mtcnn import MTCNN
from face_detector import YoloDetector
from ultraface import UltraFace


class DetectorTester:

    def __init__(self):
        self.pictures_path = "data/WIDER_train/images"
        self.folders = os.listdir(self.pictures_path)
        self.cv_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.mtcnn_detector = MTCNN()
        self.yolo_detector = YoloDetector(target_size=640, gpu=-1, min_face=90)
        self.ultra_face_detector = UltraFace()

        self.detectors = {'cv2': self.cv_detect,
                          'dlib-hog': self.dlib_detect,
                          'retina': self.retina_detect,
                          'mtcnn': self.mtcnn_detect,
                          'yolo': self.yolo_detect,
                          'ultraface': self.ultra_face_detect}

    def resize_img(self, img, x=640, y=480):
        return cv2.resize(img, (x, y), interpolation=cv2.INTER_NEAREST)

    def dlib2opencv(self, dlib_rect):
        x = dlib_rect.left()
        y = dlib_rect.top()
        w = dlib_rect.right()
        h = dlib_rect.bottom()
        return [x, y, w - x, h - y]

    def cv_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.cv_detector.detectMultiScale(gray, 1.1, 4)

    def dlib_detect(self, img):
        out = []
        for face in self.dlib_detector(img, 1):
            out.append(self.dlib2opencv(face))
        return out

    def retina_detect(self, img):
        out = []
        for face in RetinaFace.detect_faces(img).values():
            out.append(face['facial_area'])
        print(out)
        return out

    def mtcnn_detect(self, img):
        out = []
        for face in self.mtcnn_detector.detect_faces(img):
            out.append(face['box'])
        return out

    def yolo_detect(self, img):
        bbox, points = self.yolo_detector.predict(img)
        return bbox[0]

    def ultra_face_detect(self, img):
        return self.ultra_face_detector.detect(img)

    def detect(self, img, detector):
        start = time.time()
        faces = self.detectors[detector](img)
        end = time.time()
        return faces, end - start

    def prepare_pictures(self, folder):
        full_path = self.pictures_path + '/' + folder
        files = os.listdir(full_path)
        indices = random.sample(range(0, len(files)), 10)
        images = []
        for idx in indices:
            images.append(cv2.imread(full_path + '/' + files[idx]))
        return images

    def test_on_video(self, device, detector):
        cap = cv2.VideoCapture(device)
        while True:
            _, img = cap.read()
            img = self.resize_img(img)
            faces, t = self.detect(img, detector)
            for (x, y, w, h) in faces:
                if detector in ['yolo', 'retina-face', 'ultraface']:
                    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            print(t)

            cv2.imshow('img', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def test_on_pictures(self, detector):
        total_t = 0
        for image in self.prepare_pictures(self.folders[0]):
            faces, t = self.detect(image, detector)
            print(t)
            total_t += t
        print("avg:", total_t / 10)


dt = DetectorTester()
dt.test_on_video(0, 'ultraface')

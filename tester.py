import os
import time
import cv2
import dlib
import random
from retinaface import RetinaFace
from mtcnn import MTCNN
from face_detector import YoloDetector
from ultraface import UltraFace
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing as mp

def get_cpu_percent_worker(shared_cpu_samples):
    shared_cpu_samples.append(psutil.cpu_percent(1.8))


class DetectorTester:

    def __init__(self):
        self.pictures_path = "data/WIDER_train/images"
        self.folders = os.listdir(self.pictures_path)
        self.dataset_paths = []
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
        faces = RetinaFace.detect_faces(img)
        if isinstance(faces, tuple):
            return out
        for face in faces.values():
            out.append(face['facial_area'])
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

    def prepare_paths(self):
        for folder in self.folders:
            folder_paths = []
            full_path = self.pictures_path + '/' + folder
            files = os.listdir(full_path)
            indices = random.sample(range(0, len(files)), 10)
            for idx in indices:
                folder_paths.append(full_path + '/' + files[idx])
            self.dataset_paths.append(folder_paths)

    def prepare_pictures(self, folder):
        images = []
        for path in folder:
            images.append(cv2.imread(path))
        return images

    def test_on_video(self, device, detector):
        cap = cv2.VideoCapture(device)
        total_t = 0
        i = 0
        while True:
            _, img = cap.read()
            img = self.resize_img(img)
            faces, t = self.detect(img, detector)
            total_t += t
            for (x, y, w, h) in faces:
                if detector in ['yolo', 'retina', 'ultraface']:
                    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('img', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if i >= 100:
                print(f'average processing time of {detector} over {i} iterations: {total_t / i} s')
                i = 0
                total_t = 0
            i += 1

    def test_on_pictures(self, detector):
        total_t = 0
        grand_total_t = 0
        manager = mp.Manager()
        shared_cpu_samples = manager.list()
        cpu_watcher = BackgroundScheduler()
        cpu_watcher.add_job(get_cpu_percent_worker, 'interval', seconds=2, args=(shared_cpu_samples,))
        cpu_watcher.start()
        for folder_path in self.dataset_paths:
            images = self.prepare_pictures(folder_path)
            for image in images:
                faces, t = self.detect(image, detector)
                total_t += t
            grand_total_t += total_t
            print(f'avg time of {detector} over {len(images)} images: {total_t / len(images)} s')
            total_t = 0

        stat_time = f'<{detector}> number of images: {len(self.folders) * 10}, ' \
                    f'avg time: {grand_total_t / (len(self.folders) * 10)} s, ' \
                    f'avg cpu usage: {sum(shared_cpu_samples) / len(shared_cpu_samples)}% ({len(shared_cpu_samples)} samples)\n'
        print(stat_time)
        with open("stats", 'a') as file:
            file.write(stat_time)


dt = DetectorTester()
# dt.test_on_video(0, 'mtcnn')
dt.prepare_paths()
for detector in dt.detectors:
    dt.test_on_pictures(detector)

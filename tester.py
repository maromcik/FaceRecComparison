import os
import time

main_pid = os.getpid()
print("pid:", main_pid)

command = 'echo maromcik24 | sudo -S chrt -a -b -p 0 '
os.system(command + str(main_pid))

show = "chrt -p "
os.system(show + str(main_pid))

import pickle
import socket
import cv2
import dlib
import random
import numpy as np
from keras_vggface import utils
from retinaface import RetinaFace
from mtcnn import MTCNN
from yolov5 import YoloDetector
from ultraface import UltraFace
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing as mp
from arcface import ArcFace
from keras_vggface.vggface import VGGFace


def get_cpu_percent_worker(shared_cpu_samples, interval):
    shared_cpu_samples.append(psutil.cpu_percent(interval))


class DetectorTester:

    def __init__(self):
        self.pictures_path = "data/WIDER_train/images"
        self.folders = os.listdir(self.pictures_path)
        self.dataset_paths = []
        self.cv_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.mtcnn_detector = MTCNN(min_face_size=10)
        self.yolo_detector = YoloDetector(target_size=480, gpu=-1, min_face=10)
        # self.ultra_face_detector = UltraFace()
        # for RFB 640 model
        self.ultra_face_detector = UltraFace(width=640, height=480)

        self.detectors = {'cv2': self.cv_detect,
                          'dlib': self.dlib_detect,
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
        cpu_watcher.add_job(get_cpu_percent_worker, 'interval', seconds=2.2, args=(shared_cpu_samples, 2.0))
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
                    f'avg cpu usage: {sum(shared_cpu_samples) / len(shared_cpu_samples)}% ' \
                    f'({len(shared_cpu_samples)} samples)\n'
        print(stat_time)
        with open("stats", 'a') as file:
            file.write(stat_time)


class RecognitionTester:
    def __init__(self):
        self.arc_face_rec_model = ArcFace.ArcFace()
        self.dlib_face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.facenet_model = 0
        self.vgg_model = VGGFace()
        # self.openface_model = openface.TorchNeuralNet("models/openface/nn2.def.lua", 150, cuda=False)
        self.models = {"dlib": self.dlib_recognize,
                       "arcface": self.arcface_recognize,
                       "vgg": self.vgg_recognize,
                       # "openface": self.openface_recognize,
                       }
        print("initialized")

    # def detect(self, img):
    #     return self.ultra_face_detector.detect(img)
    # def to_dlib(self, face):
    #     return dlib.rectangle(face[0], face[1], face[2], face[3])
    # def dlib_align(self, image, face):
    #     # rect = self.to_dlib(face)
    #     sp = self.shape_predictor(image, face)
    #     aligned_face = dlib.get_face_chip(image, sp, 150)
    #     return aligned_face

    def vgg_recognize(self, face):
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_NEAREST)
        x = np.expand_dims(face, axis=0)
        x = x.astype('float64')
        x = utils.preprocess_input(x, version=1)  # or version=2
        return self.vgg_model.predict(x)

    def arcface_recognize(self, face):
        return self.arc_face_rec_model.calc_emb(face)

    def dlib_recognize(self, face):
        return self.dlib_face_rec_model.compute_face_descriptor(face)

    def openface_recognize(self, face):
        return self.openface_model.forward(face)

    def recognize(self, face, model):
        start = time.time()
        self.models[model](face)
        end = time.time()
        return end - start

    def test_on_pictures(self, model):
        path = "data/aligned_faces"
        faces = []
        for file in os.listdir(path):
            faces.append(cv2.imread(path + '/' + file))

        print(len(faces), "faces loaded")
        total_t = 0
        count = 0
        manager = mp.Manager()
        shared_cpu_samples = manager.list()
        cpu_watcher = BackgroundScheduler()
        cpu_watcher.add_job(get_cpu_percent_worker, 'interval', seconds=0.4, args=(shared_cpu_samples, 0.1))
        cpu_watcher.start()
        for face in faces:
            t = self.recognize(face, model)
            total_t += t
            count += 1
            print(t)

        cpu_watcher.shutdown()
        stat_time = f'<{model}> number of images: {count}, ' \
                    f'avg time: {total_t / count} s, ' \
                    f'avg cpu usage: {sum(shared_cpu_samples) / len(shared_cpu_samples)}% ' \
                    f'({len(shared_cpu_samples)} samples)\n'
        print(stat_time)
        with open("stats_face_rec", 'a') as file:
            file.write(stat_time)


class ServerLoadTesting:
    def __init__(self):

        self.dt = DetectorTester()
        self.dt.prepare_paths()
        self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.faces = []

    def to_dlib(self, face):
        return dlib.rectangle(face[0], face[1], face[2], face[3])

    def dlib_align(self, image, face):
        # rect = self.to_dlib(face)
        sp = self.shape_predictor(image, face)
        aligned_face = dlib.get_face_chip(image, sp, 150)
        return aligned_face

    def encode_image(self, image):
        img_encode = cv2.imencode('.jpg', image)[1]
        np_data = np.array(img_encode)
        byte_data = np_data.tobytes()
        return byte_data

    def send_image(self, image):
        camera_id = random.randrange(1, 6)
        camera = "{:07d}".format(camera_id)
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        addr = "127.0.0.1", 5555
        c.connect(addr)
        c.send(camera.encode())
        c.sendall(self.encode_image(image))

    def prepare_paths(self, paths):
        images = self.dt.prepare_pictures(paths)
        for image in images:
            faces, t = self.dt.detect(image, 'ultraface')
            for face in faces:
                target = self.dlib_align(image, self.to_dlib(face))
                self.faces.append(target)

    def prepare_aligned_faces(self):
        path = "data/aligned_faces"
        for file in os.listdir(path):
            self.faces.append(cv2.imread(path + '/' + file))

        print(len(self.faces), "faces loaded")

    def prepare_faces(self):
        for paths in self.dt.dataset_paths:
            self.prepare_paths(paths)
        with open('raw_faces.pkl', 'wb') as outfile:
            pickle.dump(self.faces, outfile, pickle.HIGHEST_PROTOCOL)
        print("the number of faces:", len(self.faces))

    def run_test_worker(self, faces):
        for face in faces:
            self.send_image(face)

    def load_faces(self):
        with open('raw_faces.pkl', 'rb') as infile:
            self.faces = pickle.load(infile)

    def run_test(self):
        n_faces = len(self.faces)
        print("Sending")
        p1 = mp.Process(target=self.run_test_worker, args=(self.faces[:(n_faces // 2)],))
        p2 = mp.Process(target=self.run_test_worker, args=(self.faces[(n_faces // 2):],))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        print("Finished")


if __name__ == "__main__":
    dt = DetectorTester()
    dt.prepare_paths()
    dt.test_on_pictures("ultraface")
    dt.test_on_pictures("yolo")

    # for detector in dt.detectors:
    #     dt.test_on_pictures(detector)
    #
    # rt = RecognitionTester()
    # rt.test_on_pictures('dlib')
    # for model in rt.models:
    #     rt.test_on_pictures(model)

    # slt = ServerLoadTesting()
    # slt.prepare_faces()
    # slt.load_faces()
    # start = time.time()
    # slt.run_test()
    # print("time:", time.time() - start)

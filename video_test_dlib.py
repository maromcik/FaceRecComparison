import random

import dlib
import cv2
import numpy as np
import time

class FaceRecognition:
    def __init__(self):

        # self.cap = cv2.VideoCapture("rtmp://192.168.5.55:1935/bcs/channel0_main.bcs?channel=0&stream=0&user=admin&password=123456")  # capture from camera
        self.cap = cv2.VideoCapture("oni.jpg")
        # self.cap = cv2.VideoCapture("http://192.168.59.221:8080/video")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor68 = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.facerec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.resize_factor = 0.25

    # draws the bounding rectangle to every processed frame
    def draw(self, img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # prints name of the person to every processed frame
    def PrintText(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # resizes frames
    def resize_img(self, img, fx=0.25, fy=0.25):
        return cv2.resize(img, (0, 0), fx=fx, fy=fy)

    # convers dlib coordinates to opencv coordinates
    def dlib2opencv(self, dlib_rect):
        x = dlib_rect.left()
        y = dlib_rect.top()
        w = dlib_rect.right()
        h = dlib_rect.bottom()
        return [x, y, w - x, h - y]

    # detects faces in frames
    def detect(self, img):
        n = time.time()
        faces = self.detector(img, 1)
        print("detector time: ", time.time() - n)
        if len(faces) != 0:
            return faces
        else:
            return None

    # finds landmarks in frames using the shape predictor
    def find_landmarks(self, img, faces):
        landmarks = []
        for face in faces:
            sp = self.predictor68(img, face)
            landmarks.append(dlib.get_face_chip(img, sp, size=150))
        return landmarks

    # computes descriptors
    def descriptor(self, landmarks):
        descriptors = self.facerec_model.compute_face_descriptor(landmarks)
        return np.array(descriptors)

    # compares 2 faces in 128D space
    def compare(self, known, unknown):
        return np.linalg.norm(known - unknown, axis=1)

    # reads stream from a camera, runs neccessary image processings and puts every frame to frameQ
    def read_stream(self):

        ret, frame = self.cap.read()
        if frame is not None:
            self.process(frame)


    # main function, puts everything needed for facial rec. together
    def process(self, image):
        labels = []
        # frame = self.resize_img(image, fx=self.resize_factor, fy=self.resize_factor)
        start = time.time()
        frame = image
        faces = self.detect(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if there are any faces in the frame
        if faces is not None:
            landmarks = self.find_landmarks(frame, faces)
            # for every faces do following
            for i in range(0, len(faces)):
                # convert coordinate systems
                rect = self.dlib2opencv(faces[i])
                # draw a rectangle
                self.draw(frame, rect)
                (x, y, w, h) = rect
                x = int(x * (1 / self.resize_factor))
                y = int(y * (1 / self.resize_factor))
                w = int(w * (1 / self.resize_factor))
                h = int(h * (1 / self.resize_factor))
                cv2.imwrite(f'{random.Random().randrange(1000,20000000)}.jpg', cv2.cvtColor(landmarks[i], cv2.COLOR_BGR2RGB))
                self.descriptor(landmarks[i])
                # comparisons = (self.compare(self.descriptors, self.descriptor(frame, landmarks[i]))).tolist()
                #
                #
                # if np.amin(comparisons) <= 0.55:
                #     label = np.argmin(comparisons)
                #     labels.append((label, True))
                # else:
                #     label = None
                #     labels.append(label)
                # try:
                #     self.PrintText(frame, self.names[int(label)], rect[0], rect[1])
                # except IndexError:
                #     print("Person does not exist anymore, you have most likely forgotten to load files.")
                # except TypeError:
                #     self.PrintText(frame, "unknown", rect[0], rect[1])
        end = time.time() - start
        print("PROCESS TIME: ", end)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = self.resize_img(frame)
        cv2.imshow("FaceRecognition", frame)
        cv2.waitKey(1000)




fr = FaceRecognition()
fr.read_stream()
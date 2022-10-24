import itertools
import os
import pickle
import socket
import multiprocessing as mp
import time

from apscheduler.schedulers.background import BackgroundScheduler
import dlib
import numpy as np
from django.utils import timezone
# from arcface import ArcFace

import FaceRecognition.models as database
import cv2
from django import db

resnet = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
# arcface_model = ArcFace.ArcFace()
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

model = "dlib"


def dlib_dsc(face):
    return np.array(resnet.compute_face_descriptor(face))


# def arcface_dsc(face):
#     return arcface_model.calc_emb(face)


models = {"dlib": dlib_dsc,
          # "arcface": arcface_dsc
          }


def get_descriptor(face, m):
    return models[m](face)


def dlib_compare(descriptors, dsc, threshold):
    # do for every comparison in the list comparisons
    comparisons = np.linalg.norm(descriptors - dsc, axis=1)
    for i in range(len(comparisons)):
        if comparisons[i] <= threshold:
            return True, i
    return False, -1


# def arcface_compare(descriptors, dsc, threshold):
#     # do for every comparison in the list comparisons
#     for i in range(len(descriptors)):
#         dst = arcface_model.get_distance_embeddings(descriptors[i], dsc)
#         if dst <= threshold:
#             return True, i
#     return False, -1


comparison = {"dlib": dlib_compare,
              # "arcface": arcface_compare
              }


def compare_all(descriptors, dsc, threshold, m):
    return comparison[m](descriptors, dsc, threshold)


def process_image(descriptors, staff_descriptors, staff, img):
    if len(detector(img, 1)) != 1:
        print("Invalid face in picture")
        return None, None, None
    dsc = get_descriptor(img, model)
    if staff:
        exists, idx = compare_all(staff_descriptors, dsc, 0.58, model)
        if exists:
            # this person is staff
            print("person is staff")
            return False, idx, False
    if len(descriptors) == 0:
        descriptors.append(dsc)
        # this is a new person and shared descriptors are empty
        return True, 0, True
    exists, idx = compare_all(descriptors, dsc, 0.55, model)
    if exists:
        # this is an existing person
        return True, idx, False
    descriptors.append(dsc)
    # this is a new person
    return True, len(descriptors) - 1, True


def process_connection(c, shared_descriptors, shared_staff_descriptors, person_map, staff):
    db.connections.close_all()
    # start = time.time()
    camera_id = int(c.recv(7).decode())
    # print(f"camera id: {camera_id}")
    camera = database.Camera.objects.get(pk=camera_id)
    room = camera.room
    is_entrance = camera.entrance
    is_exit = camera.exit

    fragments = []
    while True:
        chunk = c.recv(4096)
        if not chunk:
            break
        fragments.append(chunk)

    c.close()
    img = np.asarray(bytearray(b''.join(fragments)), dtype="uint8")
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    write_db, idx, new_idx = process_image(shared_descriptors, shared_staff_descriptors, staff, frame)

    if not write_db or write_db is None:
        return

    last_person = None
    last_camera = None
    if database.Log.objects.all().count() > 0 and idx in person_map:
        last_log = database.Log.objects.filter(person__id=person_map[idx]).latest('time')
        last_person = int(last_log.person.id)
        last_camera = int(last_log.camera.id)

    if write_db and last_person is not None and last_camera is not None \
            and camera_id == last_camera and person_map.get(idx, -1) == last_person:
        print(f"skipping {idx}")
    elif write_db and new_idx and is_entrance and not is_exit:
        print(f"new person with id {idx}")
        person = database.Person.objects.create()
        person_map[idx] = person.id
        database.Log.objects.create(person=person, camera=camera, room=room, time=timezone.now())
    elif write_db and not new_idx and not is_entrance and not is_exit:
        if idx in person_map:
            print(f"logging with id {idx}")
            person = database.Person.objects.get(id=person_map[idx])
            database.Log.objects.create(person=person, camera=camera, room=room, time=timezone.now())
        else:
            print(f"person {idx} already deleted")
    elif write_db and not new_idx and not is_entrance and is_exit:
        if idx in person_map:
            database.Person.objects.get(id=person_map[idx]).delete()
            del person_map[idx]
            print(len(shared_descriptors))
            del shared_descriptors[idx]
            print(len(shared_descriptors))
            room.visited += 1
            room.save()
            print("person has left the buildings")
        else:
            del shared_descriptors[idx]
            print("person has never entered the building")
    else:
        del shared_descriptors[idx]

    print(len(shared_descriptors))
    db.connections.close_all()
    # print("total time: ", time.time() - start)
    exit(0)


def load_staff_descriptors():
    with open('staff_descriptors.pkl', 'rb') as infile:
        staff_descriptors = pickle.load(infile)
    print("Staff descriptors have been loaded")
    return staff_descriptors


def prune_logs(descriptors, person_map):
    print("descriptors", len(descriptors))
    print("personmap", len(person_map), "ids:", person_map)
    i = 0
    while i < len(descriptors):
        comparisons = np.linalg.norm(descriptors - descriptors[i], axis=1)
        print(comparisons)
        for idx in range(len(comparisons) - 1, -1, -1):
            if 0.01 <= comparisons[idx] <= 0.62:
                if idx in person_map:
                    database.Person.objects.get(id=person_map[idx]).delete()
                    del person_map[idx]
                    del descriptors[idx]
                    print(f"person with {idx} deleted")
                else:
                    print(f"no such person, deleting {idx}")
                    del descriptors[idx]
        i += 1


def server_listener(s):
    manager = mp.Manager()
    shared_descriptors = manager.list()
    shared_staff_descriptors = manager.list()
    person_map = manager.dict()
    staff = True
    try:
        shared_staff_descriptors[:] = load_staff_descriptors()[:]
    except FileNotFoundError:
        print("Creating staff descriptors")
        ret = process_staff_descriptors()
        if ret == 0:
            shared_staff_descriptors[:] = load_staff_descriptors()[:]
        else:
            staff = False
            print("Staff file will not be used:")

    pruner = BackgroundScheduler()
    pruner.add_job(prune_logs, 'interval', seconds=30, args=(shared_descriptors, person_map))
    pruner.start()

    s.listen(1000)

    while True:
        c, addr = s.accept()

        # print('Connected to: ' + addr[0] + ':' + str(addr[1]))
        p = mp.Process(target=process_connection, args=(c, shared_descriptors, shared_staff_descriptors, person_map, staff),
                       daemon=True)
        p.start()

def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def process_staff_descriptors_worker(name, file):
    dir_path = os.path.join(os.path.dirname(__file__), "..") + "/media/"
    full_path = dir_path + file
    print("processing: ", full_path)
    img = load_image(full_path)
    face = detector(img, 1)
    descriptor = None
    if len(face) != 0:
        landmarks = sp(img, face[0])
        aligned_face = dlib.get_face_chip(img, landmarks, 150)
        descriptor = get_descriptor(aligned_face, model)
    else:
        print("No face in picture {}".format(full_path))
        database.Staff.objects.filter(name=name).delete()
        print("record deleted from database")
    return descriptor


def process_staff_descriptors():
    db.connections.close_all()
    staff = database.Staff.objects.all()
    names = list(staff.values_list('name', flat=True))
    files = list(staff.values_list('file', flat=True))
    pool = mp.Pool(processes=mp.cpu_count())
    descriptors = (pool.starmap(process_staff_descriptors_worker, zip(names, files)))
    descriptors = [item for item in descriptors if item is not None]
    with open('staff_descriptors.pkl', 'wb') as outfile:
        pickle.dump(descriptors, outfile, pickle.HIGHEST_PROTOCOL)
    outfile.close()
    print("descriptors of staff have been saved")

    if len(descriptors) == 0:
        print("No staff found")
        return -1
    return 0


def reset_counters():
    for room in database.Room.objects.all():
        room.visited = 0
        room.save()


def infer_ip():
    temp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    temp_s.connect(("8.8.8.8", 80))
    port = 5555
    ip = temp_s.getsockname()[0]
    return ip, port

class FaceRecognition:
    def __init__(self):

        self.addr = infer_ip()
        print("IP: ", self.addr)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # def __del__(self):
    #     for child in active_children():
    #         child.terminate()
    #     print("All children killed")
    #     self.s.close()
    #     print("Server closed")

    def run_server(self):
        self.s.bind(self.addr)

        p = mp.Process(target=server_listener, args=(self.s,))
        p.start()

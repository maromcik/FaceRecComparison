# Testing framework
## Installation
```bash
pip install -r requirements.txt
```
## Face detectors

### Description
This class was used to test the speed of execution and CPU usage of these face detectors:
* OpenCV Haar cascade feature-based face detector [available here](https://opencv.org/)
* Dlib Histogram of oriented gradients frontal face detector by [Davis King](https://github.com/davisking/dlib)
* Retina face deeply learned face detector by [serengil](https://github.com/serengil/retinaface)
* MTCNN deeply learned face detector [available here](https://pypi.org/project/mtcnn/)
* YOLO deeply learned face detector by [elyha7](https://github.com/elyha7/yoloface)
* Ultra light fast generic face detector by [Linzaer](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) for edge devices.
### Usage example
Running on a video stream from an IP camera.
```python
from tester import DetectorTester
dt = DetectorTester()
dt.test_on_video("rtmp://192.168.5.51:1935/livemain", 'yolo')
```
Running on a video stream from a directly attached or built-in webcam (try various indexes instead of 0 if it does not work).
```python
from tester import DetectorTester
dt = DetectorTester()
dt.test_on_video(0, 'yolo')
```
Testing all detectors on the WIDER dataset
```python
from tester import DetectorTester
dt = DetectorTester()
dt.prepare_paths()
for detector in dt.detectors:
    dt.test_on_pictures(detector)
```

## Face recognition models
### Description
This class was used to test the speed of execution and CPU usage of these face recognition models:
* Dlib Resent base model by [Davis King](https://github.com/davisking/dlib)
* Arcface [available here](https://pypi.org/project/arcface/)
* VGG-Face [available here](https://pypi.org/project/keras-vggface/)
### Usage example
Test all face detectors on 100 pre-processed (aligned) faces. Faces need to be pre-processed beforehand. 
```python
from tester import RecognitionTester
rt = RecognitionTester()
for model in rt.models:
    rt.test_on_pictures(model)
```

## System testing
### Description
This class was used to send pre-processed (aligned) faces to the server implementation to test the speed of execution on the server side

```python
from tester import ServerLoadTesting
slt = ServerLoadTesting()
slt.prepare_faces()
slt.load_faces()
slt.run_test()
```

# Face detector comparison

## Description
This code was used to test the speed of execution and CPU usage of these face detectors:
* OpenCV Haar cascade feature-based face detecotr
* Dlib Histogram of oriented gradients frontal face detector
* Retina face deeply learned face detector
* MTCNN deeply learned face detector
* YOLO deeply learned face detector
* Ultra ligh fast generic face detector by [Linzaer](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) for edge devices.
## Installation
```bash
pip install -r requirements.txt
```
## Usage example
Running on a video stream from an IP camera.
```python
from tester import DetectorTester
dt = DetectorTester()
dt.test_on_video("rtmp://192.168.5.51:1935/livemain", 'yolo')
```
Running on a video stream from built-in webcam.
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
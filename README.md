# Face detector comparison

## Description

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
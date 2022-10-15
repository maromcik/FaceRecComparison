import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# To capture video from webcam.
# cap = cv2.VideoCapture("rtmp://192.168.5.55:1935/bcs/channel0_main.bcs?channel=0&stream=0&user=admin&password=123456")  # capture from camera
cap = cv2.VideoCapture(2)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

def resize_img(img, fx=0.25, fy=0.25):
    return cv2.resize(img, (0, 0), fx=fx, fy=fy)


while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    # img = resize_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
# Release the VideoCapture object
cap.release()
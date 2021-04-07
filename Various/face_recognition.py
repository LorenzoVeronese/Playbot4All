import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# we want to create a list of the names of the folders of each person
p = []
for i in os.listdir(r'path of the folder'):
    p.append(i)

DIR = r'name of the folder'

features = np.load('features.npy', allow_pickle = True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'path of a validation picture')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# !!! iteration inside a rectangle
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'people = {label} with confidence = {confidence}')

    cv.putext(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 
    1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.iimshow('Detected Face', img)

cv.wait(0)
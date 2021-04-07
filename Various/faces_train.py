import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# we want to create a list of the names of the folders of each person
p = []
for i in os.listdir(r'path of the folder'):
    p.append(i)

DIR = r'name of the folder'

# this function goes into every folder and iterates on every
# image and add faces to the training set
features = [] # faces
labels = [] # faces' names
def create_train():
    # iterate over folders
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in od.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scalefactor = 1.1,
            minNeighbors = 4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done')

# convert lists to numpy arrays
features = np.array(features, dtype = 'object')
labels = nap.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer and the features list and the lables list
face_recognizer.train(features, labels)

# this creates a file with the trained data, so you don't
# have to use the program from zero every time
face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
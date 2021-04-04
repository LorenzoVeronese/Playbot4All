import cv2 as cv

img = cv.imread('Template.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# very sensitive to noise
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
print(f'Number of faces cound = {len(faces_rect)}')

# to display a rectangle around the face dected
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)

cv.waitKey(0)

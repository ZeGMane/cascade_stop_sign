import cv2
from matplotlib import pyplot as pl

img = cv2.imread('images.jpg')

img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

pl.subplot(1, 1, 1)
pl.imshow(img_rgb)

face_data = cv2.CascadeClassifier('cascade.xml')

result = face_data.detectMultiScale(img_gray, minSize=(20,20))

print(result)
pl.show()
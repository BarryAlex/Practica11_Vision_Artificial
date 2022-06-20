#Edwing Alexis Casillas Valencia.   19110113.   7E1.    Práctica 8 visión artificial
#Igualdades con rotación y reducción de fondo
from cv2 import imread
import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('pelador.jpg',1)
im2 = cv2.imread('utensilios 2.jpg',1)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

scale = 30
width = int(im1.shape[1] * scale / 100)
height = int(im1.shape[0] * scale / 100)
dsize = (width, height)
im1_n = cv2.resize(im1, dsize)
width2 = int(im2.shape[1] * scale / 100)
height2 = int(im2.shape[0] * scale / 100)
dsize2 = (width2, height2)
im2_n = cv2.resize(im2, dsize2)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(im1_n,None)
kp2, des2 = orb.detectAndCompute(im2_n,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

im3 = cv2.drawMatches(im1_n, kp1, im2_n, kp2, matches[:70], None, flags=2)
plt.imshow(im3)
plt.show()

cap = cv2.VideoCapture('VID1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)

    k=cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:12:22 2019

@author: anik
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Loading cascade for face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # Loading cascade for eye
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # Loading cascade for smile


def detect(gray, frame): #This function takes as input the image in black and white (gray) and the original image (frame). Returns the same image with rectangles
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #detectMultiScale method to locate multiple faces in the image.
    for (x, y, w, h) in faces: #For each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 250, 250), 2)
        roi_gray = gray[y:y+h, x:x+w] # Getting interested region in gray image
        roi_color = frame[y:y+h, x:x+w] # Getting interested region in color image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes: #For each detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 30)
        for (sx, sy, sw, sh) in smiles: #For each detected smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)    
    return frame


video_capture = cv2.VideoCapture(0)
while True:# We repeat infinitely (until break):
    _, frame = video_capture.read()# We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canv = detect(gray, frame)
    cv2.imshow('Video', canv)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break
video_capture.release()
cv2.destroyAllWindows()

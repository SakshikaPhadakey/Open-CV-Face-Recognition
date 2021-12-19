import numpy as np
import cv2
import pickle
import pyttsx3
import os, re
import speech_recognition as sr
import pyaudio
import time
import urllib

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

speaker = pyttsx3.init()
r = sr.Recognizer()
font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

#get labels from the dictionary using pickles
labels = {"person_name": 1}

with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = ''

while(True):
	num = 0
	# Capture frame-by-frame
	ret, frame = cap.read()

	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

	for (x, y, w, h) in faces:

		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
		roi_color = frame[y:y+h, x:x+w]
		# recognize? deep learned model predict keras tensorflow pytorch scikit learn
		id_, conf = recognizer.predict(roi_gray)
		if conf>=54 and conf <= 85:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			num = int(num) + 1
			cv2.putText(frame, name+', ID:'+str(num),(x,y), font, 1, color, stroke, cv2.LINE_AA)

		if num > 1:
			text1 = "Multiple Faces Detected, Please Speak the youtuber name you want to scan"
			speaker.say(text1)
			speaker.runAndWait()
			with sr.Microphone() as source:
				#r.pause_threshold = 1
				audio = r.listen(source)
				query = r.recognize_google(audio, language='en-in')
				print(query)

				speaker.say(query)
				speaker.runAndWait()

		img_item = "test.png"
		cv2.imwrite(img_item, roi_color)

		color = (0, 255, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

	cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)

	cv2.imshow('frame',frame)


	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

if name:
	text = f"Welcome {name} , Your are authorized"
	speaker.say(text)
else:
	text = "You are not authorized"
	speaker.say(text)
speaker.runAndWait()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

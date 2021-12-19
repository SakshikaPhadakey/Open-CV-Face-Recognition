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

text = f"Welcome Fight Club Team , Beginning the scan"
speaker.say(text)
speaker.runAndWait()
r = sr.Recognizer()
font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

#get labels from the dictionary using pickles
labels = {"person_name": 1}
detected_face= []

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
		# Create rectangle around the face
		cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

		# Recognize the face belongs to which ID
		Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
		roi_color = frame[y:y+h, x:x+w]
		# # recognize? deep learned model predict keras tensorflow pytorch scikit learn
		id_, conf = recognizer.predict(roi_gray)
		if confidence>=54 and confidence <= 85:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			num = int(num) + 1
			#cv2.putText(frame, name+', ID:'+str(num),(x,y), font, 1, color, stroke, cv2.LINE_AA)

		cv2.rectangle(frame, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
		cv2.putText(frame, name + 'ID:' +str(num), (x, y - 40), font, 1, (255, 255, 255), 3)
		detected_face.append(name)

	cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)
	cv2.imshow('frame',frame)
	cv2.waitKey(3)

	if num > 1:
		text1 = "Multiple Faces Detected, Please Speak the youtuber name you want to scan"
		speaker.say(text1)
		speaker.runAndWait()
		with sr.Microphone() as source:
			audio = r.listen(source)
			try:
				get = r.recognize_google(audio)
				speaker.say(name)
				speaker.runAndWait()

			except sr.UnknownValueError:
				print('error')
			except sr.RequestError as e:
				print('failed'.format(e))
		break
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

if name in set(detected_face):
	text = f"Welcome {name} , Your are authorized"
	speaker.say(text)
else:
	text = "You are not authorized"
	speaker.say(text)
speaker.runAndWait()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

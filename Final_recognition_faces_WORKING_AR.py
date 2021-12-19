import numpy as np
import cv2
import pickle
import pyttsx3
import speech_recognition as sr

# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

speaker = pyttsx3.init()
r = sr.Recognizer()
text = f"Welcome Fight Club Team , Beginning the scan"
speaker.say(text)
speaker.runAndWait()

#get labels from the dictionary using pickles
labels = {"person_name": 1}
detected_face = []
get = ''

with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


def implement():
	name = ''
	cap = cv2.VideoCapture(0)
	video = cv2.VideoCapture("./test.mp4")
	while(True):
		num = 0
	# Capture frame-by-frame
		ret, frame = cap.read()
		ret, source = video.read()
		font = simplex_font()

		faces = facial_detection(frame)

		for (x, y, w, h) in faces:
			# Create rectangle around the face
			cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
			gray_roi = roi_gray(frame,x,y,w,h)
			roi = roi_color(frame,x,y,w,h)
			id_, conf = predict(gray_roi)

			if conf>=54 and conf <= 85:
				name = labels[id_]
				# color = (255, 255, 255)
				# stroke = 2
				#cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
				num = int(num) + 1
				#cv2.putText(frame, name+', ID:'+str(num),(x,y), font, 1, color, stroke, cv2.LINE_AA)
				warped = display_video(frame,source)
				if warped is not None:
					frame = warped

			cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
			cv2.rectangle(frame, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)

			cv2.putText(frame, name, (x, y - 40), font, 1, (255, 255, 255), 3)
			detected_face.append(name.replace(' ', '').replace('-', '').lower())
			cv2.waitKey(2)
		cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)
		cv2.imshow('frame', frame)
		cv2.waitKey(5)

		if num > 1:
			text1 = "Multiple Faces Detected, Please Speak the youtuber name you want to scan"
			speaker.say(text1)
			speaker.runAndWait()
			with sr.Microphone() as source:
				audio = r.listen(source)
				try:
					get = r.recognize_google(audio)
					speaker.say(get)
					print(get)
					speaker.runAndWait()
					break
				except sr.UnknownValueError:
					print('error')
				except sr.RequestError as e:
					print('failed'.format(e))
		if cv2.waitKey(20) & 0xFF == ord('q'):
			get = name
			break

	if get.replace(' ','').replace('-','').lower() in set(detected_face):
		text = f"Welcome {get} , Your are authorized"
		speaker.say(text)
	else:
		text = "You are not authorized"
		speaker.say(text)
	speaker.runAndWait()

# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def facial_detection(frame):
	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
	faces = face_cascade.detectMultiScale(gray(frame), scaleFactor=1.5, minNeighbors=5)
	return faces

def gray(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def roi_gray(frame,x,y,w,h):
	gray_frame = gray(frame)
	return gray_frame[y:y+h, x:x+w]

def roi_color(frame,x,y,w,h):
	return frame[y:y+h, x:x+w]

def predict(roi_gray):
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("./recognizers/face-trainner.yml")
	return recognizer.predict(roi_gray)

def simplex_font():
	return cv2.FONT_HERSHEY_SIMPLEX

def display_video(frame, src):
	(imgH, imgW) = frame.shape[:2]
	(srcH, srcW) = src.shape[:2]

	srcMat = sourceMatrix(srcW,srcH)
	dstMat = destinationMatrix()

	(H, _) = calculate_homography(srcMat, dstMat)
	warped = cv2.warpPerspective(src, H, (imgW, imgH))
	mask = np.zeros((imgH, imgW), dtype="uint8")
	cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),cv2.LINE_AA)
	maskScaled = mask.copy() / 255.0
	maskScaled = np.dstack([maskScaled] * 3)
	warpedMultiplied = cv2.multiply(warped.astype("float"),maskScaled)
	imageMultiplied = cv2.multiply(frame.astype(float),1.0 - maskScaled)
	output = cv2.add(warpedMultiplied, imageMultiplied)
	output = output.astype("uint8")
	return output

def calculate_homography(srcMat,dstMat):
	return cv2.findHomography(srcMat, dstMat)

def sourceMatrix(srcW,srcH):
	return np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

def destinationMatrix():
	return np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

implement()
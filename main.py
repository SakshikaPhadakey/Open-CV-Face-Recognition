# Modules
import cv2
import numpy as np
import pickle
import pyttsx3
import speech_recognition as sr
import arucoFunctionality
speaker = pyttsx3.init()
r = sr.Recognizer()

# Main Function.
def init():
    run_speech("Welcome Fight Club Team , Beginning the scan")
    labels = load_pickle_data()
    frame(labels)
    return


def frame(labels):
    web_cam = cv2.VideoCapture(0)
    videos = capture_all_videos()
    
    while True:
        num = 0
        single_user(web_cam,labels,num,videos)

        if cv2.waitKey(1) == ord('q'):
            break

def single_user(web_cam_video,labels,num,videos):
    ret, frame = web_cam_video.read()
    faces = facial_detection(frame)
    name = ''
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in faces:
        gray_roi = roi_gray(frame,x,y,w,h)
        id_, conf = predict(gray_roi)
        name = labels[id_]
        cv2.putText(frame,name, (x,y+h),font, 1, (255, 255, 255), 3)
    
    if(name):
        # video = cv2.VideoCapture("./video/" + name + ".mp4")
        ret, source = videos[name].read()
        num = int(num) + 1
        
        if source is not None:
            warped = find_and_warp(
                frame, 
                source,
                cornerIds=(1, 2, 4, 3)
            )

        if warped is not None:
            frame = warped

    cv2.imshow('Augmented Reality', frame)
    cv2.waitKey(4)
    return num

def capture_all_videos():
    return {
        'bharti-singh' : cv2.VideoCapture("./video/bharti-singh.mp4"),
        'malvika-sitlani' : cv2.VideoCapture("./video/malvika-sitlani.mp4"),
        'gaurav-taneja' : cv2.VideoCapture("./video/gaurav-taneja.mp4"),
        'bhuvan-bam': cv2.VideoCapture("./video/bhuvan-bam.mp4")
    }
    
def multiple_users():
    run_speech("Multiple faces detected, Your Favourite youtubers please !")
    with sr.Microphone() as source:
        audio = r.listen(source)
        get = r.recognize_google(audio)
        speaker.say(get)
        print(get)
        speaker.runAndWait()
    return 
            
def find_and_warp(frame, source, cornerIds):
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]
    markerCorners, markerIds, rejectedCandidates = arucoFunctionality.detect_markers(frame)
    if len(markerCorners)!= 4:
        markerIds = np.array([]) 
    else: 
        markerIds.flatten()
    refPts = []
    for i in cornerIds:
        j = np.squeeze(np.where(markerIds == i))
        if j.size == 0:
            continue
        else:
            j = j[0]   

        markerCorners = np.array(markerCorners)
        corner = np.squeeze(markerCorners[j])
        refPts.append(corner)
    
    if len(refPts) != 4:
            return None
    (refPtTL, refPtTR, refPtBR, refPtBL) = np.array(refPts)
    print("dst mat",refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3])
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
        cv2.LINE_AA)
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    warpedMultiplied = cv2.multiply(warped.astype("float"),
        maskScaled)
    imageMultiplied = cv2.multiply(frame.astype(float),
        1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    return output

def facial_detection(frame):
	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
	faces = face_cascade.detectMultiScale(gray(frame), scaleFactor=1.5, minNeighbors=5)
	return faces

def gray(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def roi_gray(frame,x,y,w,h):
	gray_frame = gray(frame)
	return gray_frame[y:y+h, x:x+w]


def predict(roi_gray):
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("./recognizers/face-trainner.yml")
	return recognizer.predict(roi_gray)

def run_speech(text):
    speaker.say(text)
    speaker.runAndWait()
    return

def load_pickle_data():
    labels = {}
    with open("pickles/face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
    
    return labels

init()

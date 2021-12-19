import cv2
import numpy as np
import pickle

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

labels = {"person_name": 1}

with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
# Load the predefined dictionary

def createAruco():

    # Generate the marker
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)
    cv2.imwrite("marker33.png", markerImage)
    return 

def detectingArucoMarkers():
    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()

    frame(parameters)
    # Detect the markers in the image
    return 


def frame(params):
    videostream = cv2.VideoCapture(0)
    
    while True:
        ret, frame = videostream.read()
        faces = facial_detection(frame)
        
        for (x, y, w, h) in faces:
            gray_roi = roi_gray(frame,x,y,w,h)
            id_, conf = predict(gray_roi)
            name = labels[id_]
            
            if(name):
                print("name is",name)
                video = cv2.VideoCapture("./video/" + name + ".mp4")
                ret, source = video.read()

                if source is not None:
                    warped = find_and_warp(
                        frame, source,
                        cornerIds=(1, 2, 4, 3),
                        dictionary=dictionary,
                        parameters=params,
                    )

                    if warped is not None:
                        frame = warped
                        break

        cv2.imshow('Augmented Reality', frame)
        if cv2.waitKey(1) == ord('q'):
            break

def find_and_warp(frame, source, cornerIds, dictionary, parameters):
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if len(markerCorners)!= 4:
        markerIds = np.array([]) 
    else: 
        markerIds.flatten()
    refPts = []
    for i in cornerIds:
        print("cornerIds is",cornerIds)
        j = np.squeeze(np.where(markerIds == i))
        if j.size == 0:
            continue
        else:
            j = j[0]   

        markerCorners = np.array(markerCorners)
        #print(markerCorners)
        corner = np.squeeze(markerCorners[j])
        refPts.append(corner)
    
    print("refPts",refPts)
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

detectingArucoMarkers()

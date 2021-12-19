import cv2
import numpy as np

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
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
    video = cv2.VideoCapture("./video/bharti-singh.mp4")
    
    while True:

        ret, frame = videostream.read()
        ret, source = video.read()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # scale_percent = (1000/frame.shape[0])*100
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # width2 = int(source.shape[1] * scale_percent / 100)
        # height2 = int(source.shape[0] * scale_percent / 100)
        # dim = (width2, height2)
        # source = cv2.resize(source, dim, interpolation = cv2.INTER_AREA)
    
        warped = find_and_warp(
            frame, source,
            cornerIds=(1, 2, 4, 3),
            dictionary=dictionary,
            parameters=params,
        )
        if warped is not None:
            frame = warped

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

detectingArucoMarkers()
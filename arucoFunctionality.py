# Modules
import cv2

# Get Aruco markers dictionary and parameters
def load_initialize_aruco():
    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()

    return {
        "dictionary": dictionary,
        "params": parameters
    }

def detect_markers(frame):
    aruco_details = load_initialize_aruco()
    return cv2.aruco.detectMarkers(frame, aruco_details['dictionary'], parameters=aruco_details['params'])
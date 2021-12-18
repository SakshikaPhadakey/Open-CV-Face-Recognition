import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite("images/sakshika-phadakey/img%d.jpg"% count, frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    count += 1
    if cv2.waitKey(20) & 0xFF == ord('q') or count>20:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
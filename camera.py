import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret,frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    if key == ord(' '):
        cv2.imwrite('./static/img/capture.jpg',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 


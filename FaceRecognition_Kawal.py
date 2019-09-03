import cv2
import numpy as np

# Loading the cascades
face_cascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Function that will do the detections 
def detect(gray , color):
    faces = face_cascade.detectMultiScale(gray , 1.3 , 5)
    for (x , y , w , h) in faces:
        cv2.rectangle(color , (x , y) , (x + w , y + h) , (255 , 255 , 0) ,  2 )
        eye_scan_area_gray  =  gray[y:y+h , x:x+w]
        eye_scan_area_color = color[y:y+h , x:x+h]
        eyes = eye_cascade.detectMultiScale(eye_scan_area_gray , 1.1 , 5)
        for(ex , ey , ew , eh) in eyes:
            cv2.rectangle(eye_scan_area_color , (ex , ey) , (ex + ew , ey + eh) , (0 , 255 , 255) , 2)
        
        smile = smile_cascade.detectMultiScale(eye_scan_area_gray , 1.3 , 50 )
        for(sx , sy , sw , sh) in smile:
            cv2.rectangle(eye_scan_area_color , (sx , sy) , ( sx + sw , sy + sh ) , (255 , 0 , 255) , 2)
    
    return color

video_capture = cv2.VideoCapture(0)
while True:
    ret , color = video_capture.read()
    gray = cv2.cvtColor(color , cv2.COLOR_BGR2GRAY )
    canvas = detect(gray , color)
    cv2.imshow("Video" , color)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
    



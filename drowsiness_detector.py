from playsound import playsound
import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import winsound

close_eye_size=215

alarm_time=30

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
#Detect face & eyes
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Run the cam.
print("starting video stream thread...")
vs = VideoStream(src='rtsp://admin:Ntd78952741862@192.168.1.103:554').start()
time.sleep(1.0)

num=0 #Count time when eyes close
num1=0 #Count time without face

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 800)
    L, gray = lr.light_removing(frame)
    rects = detector(gray,0)
    #Nếu không xuất hiện khuôn mặt trong quá trình lấy xe thì báo
    if len(rects)==0:
        num1=num1+1
    else:
        num1=0
    if num1>alarm_time:
            #playsound('power_alarm.wav')
            # winsound.PlaySound('power_alarm.wav', winsound.SND_FILENAME|winsound.SND_NOWAIT)
            cv2.putText(frame, "DROWSINESS ALARM!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,30,255), 2)

    #Detect khuông mặt
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #Detect mắt    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
        both_ear = (leftEAR + rightEAR) * 500  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        
        #Nếu mắt nhỏ hơn 1 giá trị close_eye_size thì báo
        if both_ear<=close_eye_size:
            num=num+1
        else:
            num=0
        if num>alarm_time:
            #playsound('power_alarm.wav')
            # winsound.PlaySound('power_alarm.wav', winsound.SND_ASYNC | winsound.SND_ALIAS )
            cv2.putText(frame, "DROWSINESS ALARM!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,30,255), 2)


    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
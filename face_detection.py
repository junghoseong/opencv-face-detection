import numpy as np
import cv2

xml = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)
#opencv 깔려있는 경로에 있는 내장함수 face detection용 xml

cap = cv2.VideoCapture(0)
#노트북 내장 카메라 인식
cap.set(3, 1200)
cap.set(4, 800)
#창의 크기는 1200*800의 크기로 만든다.

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray scale로 다시 변환해서 가져온다.
    
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    #얼굴들 인식 5개까지 해준다!, faces 변수 안에는 x, y, w, h가 리스트 형태로 들어있다.
    
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5) #image에 좌표값을 입력해서 네모박스 만든다. 빨간색 두께 5로 만든다.
    
    cv2.imshow('result', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        #ESC키 누르면 종료되도록
cap.release()
cv2.destroyAllWindows()
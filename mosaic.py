import numpy as np
import cv2

xml = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)
#opencv 깔려있는 경로에 있는 내장함수 face detection용 xml

cap = cv2.VideoCapture(0)
#노트북 내장 카메라 인식
cap.set(3, 640)
cap.set(4, 480)
#창의 크기는 640*480의 크기로 만든다.

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray scale로 다시 변환해서 가져온다.
    
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    #얼굴들 인식 5개까지 해준다!, faces 변수 안에는 x, y, w, h가 리스트 형태로 들어있다.
    
    if len(faces):
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w] #인식된 얼굴 이미지를 저장
            face_img = cv2.resize(face_img, dsize=(0,0), fx=0.04, fy=0.04) #저장한 이미지의 높이/너비를 0.04배로 조정한다.
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) #다시 원래 비율로 되돌려준다. interpolation과정에서 이미지의 해상도 왜곡이 발생한다 (모자이크)
            frame[y:y+h, x:x+w] = face_img #이제 그 영역을 바꿔준다.
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5) #image에 좌표값을 입력해서 네모박스 만든다. 빨간색 두께 5로 만든다.
    
    cv2.imshow('result', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        #ESC키 누르면 종료되도록
cap.release()
cv2.destroyAllWindows()

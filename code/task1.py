print("正在初始化摄像头...")
import cv2
import os
import datetime
cap = cv2.VideoCapture(0)

name="liuyanlin"

savedpath =r'../faceImages/' + name
isExists = os.path.exists(savedpath)
if not isExists:
    os.makedirs(savedpath)
    print('path of %s is build' % (savedpath))
else:
  print('path of %s already exist and rebuild' % (savedpath))
print("按N键拍摄图片")
i=0

while(i < 599):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, 1)
    cv2.imshow('test',frame)
    # now = datetime.datetime.now()
    # now = now.strftime('%m-%d-%H-%M-%S')
    savedname = '/'+ str(i) + '.jpg'
    cv2.imwrite(savedpath + savedname, frame)
    i += 1
  
cap.release()
cv2.destroyAllWindows()
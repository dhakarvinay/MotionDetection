#import opencv- pyhton module
import cv2

#capture live stream
cap = cv2.VideoCapture(0)

#create MOG2 background subtractor object
mog = cv2.createBackgroundSubtractorMOG2()



while True:
    #read frame
    ret,frame = cap.read()

    #convert frame to gray scale 
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #apply background subtractor
    fgmask = mog.apply(gray)
    #remove noise by morphological transformation and also fill the gap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fgmask=cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    
    #capture contours
    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #capture bigger contours
    for contour in contours:
        #ignore small contour
        if cv2.contourArea(contour)<1000:
            continue
        #then draw bounding rectangle
        x,y,w,h=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w , y+h),(0,255,0),2)

    cv2.imshow('Motion Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 #release and destroy 
cap.release()
cv2.destroyAllWindow()
print("camera released succesfully")
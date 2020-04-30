import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')

must_img = cv2.imread('mustache.png')
#must_img = cv2.cvtColor(must_img,cv2.COLOR_BGR2RGBA)

while True:
	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		noses = nose_cascade.detectMultiScale(frame,1.3,5)

		for nose in noses:
			xn,yn,wn,hn = nose
			#cv2.rectangle(frame,(xn,yn),(xn+wn,yn+hn),(0,255,0),2)
			
			nose_centre = ( int(xn+wn/2),int(yn+hn/2) )
			nose_centre_bottom = ( nose_centre[0],int(nose_centre[1]+hn/2) )

			must_start = (xn,yn+hn)
			must_end = (xn+wn,int(yn+hn+hn/2))

			must_width = wn
			must_height = int(wn/2)

			must = cv2.resize(must_img,(must_width,must_height))
			#must_gray = cv2.cvtColor(must,cv2.COLOR_BGR2GRAY)

			#must_area = frame[ must_start[1]:must_start[1]+must_height , must_start[0]:must_start[0]+must_width ]
			
			#_,must_mask = cv2.threshold(must_gray,25,255,cv2.THRESH_BINARY_INV)

			#must_area_no_must = cv2.bitwise_and(must_area,must_area,mask = must_mask)
			#final_must = cv2.add(must_area_no_must,must)

			frame[ must_start[1]:must_start[1]+must_height , must_start[0]:must_start[0]+must_width ] = must
			#cv2.rectangle(frame,must_start,must_end,(0,255,255),2)
			#cv2.circle(frame,nose_centre_bottom,3,(255,0,0),-1)
			#cv2.imshow('Frame',must_area_no_must)
			#cv2.imshow('mustache',must)
			#cv2.imshow('must',final_must)
			
		cv2.imshow('frame',frame)

	key = cv2.waitKey(1)

	if key == 27:
		break
import cv2
import numpy as np
import pytesseract as pyt

#Adding tesseract to path
pyt.pytesseract.tesseract_cmd= r"E:\Pytesseract\tesseract.exe"

#Cascade classifier
cascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

#Creating a function to read and process the image
def Read(image_name):
    img= image_name
    hImg= img.shape[1]
    WImg= img.shape[0]
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Nplate= cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in Nplate:
        #Finding the number plate from the image
        a= int(0.05*img.shape[0])
        b= int(0.06*img.shape[1])
        plate= img[y+a: y+a+h, x+b: x+b+w]
        img= cv2.rectangle(img, (x+b, y+a), (x+b+w, y+a+h), (0,0,255) , 2)

        #Pre filtering
        kernel= np.ones((1,1), np.uint8)
        plate= cv2.dilate(plate, kernel, iterations= 2)
        plate= cv2.erode(plate, kernel, iterations= 2)
        plate_gray= cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh, plate= cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow("plate", plate)

        #Text detection using Pytesserct
        text= pyt.image_to_string(plate)
        print(text)

        #Putting the text recieved on the image
        img= cv2.putText(img, text, (x+b, y+a), 
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,50), 2)



img= cv2.imread("image4.jpg")
Read(img)
cv2.imshow("Car", img)
cv2.waitKey(0)
cv2.destroyAllWindows

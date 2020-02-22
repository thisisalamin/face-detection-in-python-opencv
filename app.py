import cv2 as cv

#Casecade
face_cascade = cv.CascadeClassifier("src/haarcascade_frontalface_default.xml")


#Read Image
img = cv.imread("src/example.jpg")

#Read Image as Gray Scale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Search the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
	img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)

cv.imshow('Front Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
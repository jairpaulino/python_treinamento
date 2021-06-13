import cv2

classificadorFace = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\pessoas\\haarcascade_eye.xml')


img = cv2.imread('C:\\Users\\jairp\\Desktop\\pessoas\\pessoas3.jpg')
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = classificadorFace.detectMultiScale(imgCinza,
                                           scaleFactor = 1.2,
                                           minNeighbors = 3,
                                           minSize = (50, 50))

for (x, y, l, a) in faces:
    img = cv2.rectangle(img, (x, y), (x+l, y+a), (0, 0, 205), thickness = 3)
    regiao = img[y:y+a, x:x+l]
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhos = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, 
                                                scaleFactor = 1.005,
                                                minNeighbors = 5)
    for (x, y, l, a) in olhos:
        imgOlho = cv2.rectangle(regiao, (x, y), (x+l, y+a), (0, 255, 255), 3)

    
    
cv2.imshow("Faced e olhos", img)
cv2.waitKey()
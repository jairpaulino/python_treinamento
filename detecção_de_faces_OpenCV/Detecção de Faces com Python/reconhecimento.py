import cv2

#classificadorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classificadorFace = cv2.CascadeClassifier('cars.xml')
classificadorOlho = cv2.CascadeClassifier('haarcascade_eye.xml')

imagem = cv2.imread('carro1.jpg')

facesDetectadas = classificadorFace.detectMultiScale(imagem, scaleFactor=1.4, minNeighbors=1)
#facesDetectadas = classificadorFace.detectMultiScale(imagem, scaleFactor=1.08, minNeighbors=10, minSize=(30, 30))

for (x, y, w, h) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 5)
'''
    imagem = cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = imagem[y:y + h, x:x + w]
    roi_color = imagem[y:y + h, x:x + w]
    eyes = classificadorOlho.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
'''

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey(0)
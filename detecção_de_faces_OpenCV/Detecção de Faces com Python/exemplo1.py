import cv2

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas\\pessoas1.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificadorFace.detectMultiScale(imagemCinza)
print(facesDetectadas)
print(len(facesDetectadas))

for (x, y, w, h) in facesDetectadas:
    print(x, y, w, h)
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey(0)
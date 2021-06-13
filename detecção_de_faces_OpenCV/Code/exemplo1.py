import cv2

classificador = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\haarcascade_frontalface_default.xml')

imagem = cv2.imread('C:\\Users\\jairp\\Desktop\\pessoas\\pp2.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificador.detectMultiScale(imagemCinza, 
                                                 scaleFactor = 1.2,
                                                 minNeighbors = 2,
                                                 minSize = (50, 50))

print(len(facesDetectadas))
print(facesDetectadas)

for (x, y, l, a) in facesDetectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x,y), (x+l,y+a), (0, 0, 255, 2), thickness = 3)

cv2.imshow("Faces detectadas", imagem)
cv2.waitKey()
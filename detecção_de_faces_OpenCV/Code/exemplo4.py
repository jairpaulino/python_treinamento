import cv2

#classificador = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\haarcascade_frontalcatface.xml')
#classificador = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\relogios.xml')
classificador = cv2.CascadeClassifier('C:\\Users\\jairp\\Desktop\\cars.xml')


imagem = cv2.imread('C:\\Users\\jairp\\Desktop\\carro3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectado = classificador.detectMultiScale(imagemCinza,
                                               scaleFactor = 1.01,
                                               minNeighbors = 10)

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x+l, y+a), (0,0,255), 2)



cv2.imshow('Objetos encontrados: '+ str(len(detectado)), imagem)
cv2.waitKey()
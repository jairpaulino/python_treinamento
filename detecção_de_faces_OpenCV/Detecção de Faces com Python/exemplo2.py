import cv2

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

imagem = cv2.imread('pessoas\\faceolho.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificadorFace.detectMultiScale(imagemCinza)

for (x, y, w, h) in facesDetectadas:
    imagem = cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)
    regiao = imagem[y:y + h, x:x + w]
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
    for (ox, oy, ow, oh) in olhosDetectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)

cv2.imshow("Faces e olhos encontrados", imagem)
cv2.waitKey(0)
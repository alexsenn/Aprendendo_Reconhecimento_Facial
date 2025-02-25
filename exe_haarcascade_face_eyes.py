import cv2  # OpenCV
import numpy as np

## ----------------------------------------------------------------##
## Carregar a imagem 
imagem = cv2.imread('Arquivos/images/person.jpg')

## ----------------------------------------------------------------##
## Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

## ----------------------------------------------------------------##
## Carrega o classificador Haar Cascade 
detector_face = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_frontalface_default.xml')
detector_eyes = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_eye.xml')

## ----------------------------------------------------------------##
## Detecta eyes
deteccoes_face = detector_face.detectMultiScale(imagem_cinza)
deteccoes_eyes = detector_eyes.detectMultiScale(imagem_cinza, minNeighbors=10, minSize = (30,30) )

## ----------------------------------------------------------------##
## Verifica as dimensões da imagem
# print(imagem_cinza.shape)

## ----------------------------------------------------------------##
## Exibir retangulos
for (x, y, w, h) in deteccoes_face:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

for (x, y, w, h) in deteccoes_eyes:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0,255), 2)

## ----------------------------------------------------------------##
## Exibe as imagens
cv2.imshow('Imagem', imagem)

## ----------------------------------------------------------------##
## Aguarda uma tecla ser pressionada
cv2.waitKey(0)  
cv2.destroyAllWindows()  # Fecha a janela
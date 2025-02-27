import cv2  # OpenCV
import numpy as np

## ----------------------------------------------------------------##
## Carregar a imagem 
imagem = cv2.imread('Arquivos/images/smile.jpg')

## ----------------------------------------------------------------##
## Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

## ----------------------------------------------------------------##
## Carrega o classificador Haar Cascade smile
detector_smile = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_smile.xml')

## ----------------------------------------------------------------##
## Detecta smile
deteccoes = detector_smile.detectMultiScale(imagem_cinza, scaleFactor = 1.25, minNeighbors=5)

## ----------------------------------------------------------------##
## Verifica as dimensões da imagem
print(imagem_cinza.shape)

## ----------------------------------------------------------------##
## Exibir retangulos
for (x, y, w, h) in deteccoes:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,255,0), 2)

## ----------------------------------------------------------------##
## Exibe as imagens
cv2.imshow('Imagem', imagem)

## ----------------------------------------------------------------##
## Aguarda uma tecla ser pressionada
cv2.waitKey(0)  
cv2.destroyAllWindows()  # Fecha a janela
import cv2  # OpenCV
import numpy as np

## ----------------------------------------------------------------##
## Carregar a imagem 
imagem = cv2.imread('Arquivos/images/people3.jpg')

## ----------------------------------------------------------------##
## Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

## ----------------------------------------------------------------##
## Carrega o classificador Haar Cascade
detector_facial = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_frontalface_default.xml')

## ----------------------------------------------------------------##
## Detecta faces, aplicado ScaleFactor para especificar quanto o tamanho da imagem é reduzida
# deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor = 1.2)

## minNeighbors que controla quantos vizinhos cada janela deve ter para que a área da imagem seja considerada uma face.
# deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor = 1.2, minNeighbors = 4)

## minSize
# deteccoes = detector_facial.detectMultiScale(imagem_cinza, minSize = (76,76))

## maxSize
# deteccoes = detector_facial.detectMultiScale(imagem_cinza, maxSize = (70,70))

## Exemplo
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor = 1.1, minSize=(50,50), minNeighbors=2)

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
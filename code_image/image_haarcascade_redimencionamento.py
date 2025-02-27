import cv2  # OpenCV
import numpy as np

## Verifica a versão do OpenCV
# print(cv2.__version__)

## ----------------------------------------------------------------##
## Carregar a imagem 
# imagem = cv2.imread('Arquivos/images/person.jpg')
imagem = cv2.imread('Arquivos/images/people1.jpg')

## ----------------------------------------------------------------##
## Redimencionamento - Manual
# nova_largura = 600
# proporcao = 1680 / 1120 
# nova_altura = int(nova_largura / proporcao)
# imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))

## ----------------------------------------------------------------##
## Redimencionamento - Escala
imagem = cv2.resize(imagem, (0,0), fx = 0.5, fy = 0.5)

## ----------------------------------------------------------------##
## Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

## ----------------------------------------------------------------##
## Carrega o classificador Haar Cascade
detector_facial = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_frontalface_default.xml')

## ----------------------------------------------------------------##
## Detecta faces
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
# print(deteccoes) #Posicoes
# print(len(deteccoes))

## ----------------------------------------------------------------##
## Verifica as dimensões da imagem
# print(imagem.shape)
# print(imagem_cinza.shape)
# print(imagem_redimensionada.shape)


## ----------------------------------------------------------------##
## Exibir retangulos
for (x, y, w, h) in deteccoes:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,255,255), 3)

## ----------------------------------------------------------------##
## Exibe as imagens
cv2.imshow('Imagem', imagem)
# cv2.imshow('Imagem Cinza', imagem_cinza)
# cv2.imshow('Imagem', imagem_redimensionada)

## ----------------------------------------------------------------##
## Aguarda uma tecla ser pressionada
cv2.waitKey(0)  
cv2.destroyAllWindows()  # Fecha a janela
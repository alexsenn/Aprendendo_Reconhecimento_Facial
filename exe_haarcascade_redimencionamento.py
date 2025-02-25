import cv2  # OpenCV
import numpy as np

# Verifica a versão do OpenCV
# print(cv2.__version__)

# Carrega a imagem 
# imagem = cv2.imread('Arquivos/images/person.jpg')
imagem = cv2.imread('Arquivos/images/people1.jpg')

# Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Carrega o classificador Haar Cascade
detector_facial = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_frontalface_default.xml')

# Detecta faces
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
print(deteccoes)

# Verifica as dimensões da imagem
# print(imagem.shape)
# print(imagem_cinza.shape)

for (x, y, w, h) in deteccoes:
  #print(x, y, w, h)
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 4)
cv2.imshow('Imagem', imagem)

# Exibe as imagens
# cv2.imshow('Imagem', imagem)
# cv2.imshow('Imagem Cinza', imagem_cinza)
cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
cv2.destroyAllWindows()  # Fecha a janela
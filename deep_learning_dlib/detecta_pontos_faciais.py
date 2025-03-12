import cv2
import numpy as np
from PIL import Image
import zipfile
import os
import dlib

detector_face = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')

imagem = cv2.imread('Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/Pessoa.005.E(6).jpg')
deteccoes_faces = detector_face(imagem, 1)
for face in deteccoes_faces:
  pontos = detector_pontos(imagem, face)
  for ponto in pontos.parts():
    cv2.circle(imagem, (ponto.x, ponto.y), 2, (0,255,0), 1)

  print(len(pontos.parts()), pontos.parts())

  l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
  cv2.rectangle(imagem, (l, t), (r, b), (0,255,255), 2)

cv2.imshow('Imagem de teste',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
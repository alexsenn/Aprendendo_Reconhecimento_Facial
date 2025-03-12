import cv2
import numpy as np
from PIL import Image
import zipfile
import os
import dlib

# Inicializa os detectores e extratores do dlib
detector_face = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')
extrator_descritor_facial = dlib.face_recognition_model_v1('Arquivos/weights/dlib_face_recognition_resnet_model_v1.dat')

# Carrega a imagem com cv2 (corrige o caminho)
imagem_teste_path = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/Pessoa.007.G(4).jpg'
imagem_np = cv2.imread(imagem_teste_path)  # Lê como BGR

# Verifica se a imagem foi carregada corretamente
if imagem_np is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem: {imagem_teste_path}")

# Converte diretamente de BGR (cv2) para escala de cinza
imagem_gray = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)

# Converte para RGB para o dlib (se necessário, já que dlib espera RGB)
imagem_rgb = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2RGB)

# Detecta faces
deteccoes_faces = detector_face(imagem_rgb, 1)
for face in deteccoes_faces:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem_rgb, (l, t), (r, b), (0, 0, 255), 2)

    # Detecta pontos faciais
    pontos = detector_pontos(imagem_rgb, face)
    for ponto in pontos.parts():
        cv2.circle(imagem_rgb, (ponto.x, ponto.y), 2, (0, 255, 0), 1)

    # Extrai o descritor facial
    descritor_face = extrator_descritor_facial.compute_face_descriptor(imagem_rgb, pontos)
    descritor_face = [f for f in descritor_face]  # Converte para lista
    descritor_face = np.asarray(descritor_face, dtype=np.float64)  # Converte para array NumPy
    descritor_face = descritor_face[np.newaxis, :]  # Adiciona uma dimensão

    print(descritor_face.shape)  # Deve ser (1, 128)
    print(descritor_face)

# Converte de volta para BGR para exibição com OpenCV
imagem_bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)

# Exibe a imagem processada
cv2.imshow('Imagem de teste', imagem_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2  # OpenCV
# import matplotlib.pyplot as plt
import numpy as np
import time

## ----------------------------------------------------------------##
## Carregar o video 
arquivo_video = 'Arquivos/videos/video01.mp4'
cap = cv2.VideoCapture(arquivo_video)

## ----------------------------------------------------------------##
## Leitura de cada frame do video
conectado, video = cap.read()
print(conectado)

# ## ----------------------------------------------------------------##
# ## Verifica as dimensões do video
print(video.shape)

video_largura = video.shape[1]
video_altura = video.shape[0]
print(video_largura, video_altura)

# ## ----------------------------------------------------------------##
# ## Redimencionamento do video
largura_maxima = 900

# ## ----------------------------------------------------------------##
# ## Função para fazer os calculos da nova altura e largura
def redimenciona_video(largura, altura, largura_maxima = 600):
     if largura > largura_maxima :
          proporcao = largura / altura
          video_largura = largura_maxima
          video_altura = int(video_largura / proporcao)
     else :
       video_largura = largura
       video_altura = altura
     return video_largura, video_altura

if largura_maxima is not None:
    video_largura, video_altura = redimenciona_video(video_largura, video_altura, largura_maxima)          
print(video_largura, video_altura)

# ## ----------------------------------------------------------------##
# ## Processamento do video e captura do resultado
arquivo_resultado = 'resultado.avi'

# ## ----------------------------------------------------------------##
# ## Gravando o video na extenção 'XVID'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# ## ----------------------------------------------------------------##
# ## Definindo a velocidade do video
fps = 24

# ## ----------------------------------------------------------------##
# ## Escrever novo video
video_saida = cv2.VideoWriter(arquivo_resultado, fourcc, fps, (video_largura, video_altura))

# ## ----------------------------------------------------------------##
# ## Criando o classificador
detector_face_haar = cv2.CascadeClassifier('Arquivos\cascades\haarcascade_frontalface_default.xml')

def detecta_face_haar(detector_face, imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor = 1.15, minNeighbors = 5, minSize = (50,50))
    for (x, y, w, h) in deteccoes:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return imagem

# ## ----------------------------------------------------------------##
# ## Processamento do video frame by frame
frames_show = 200
frame_atual = 1
max_frames = -1

while cv2.waitKey(1) < 0:
    conectado, frame = cap.read()
    if not conectado:
        break
    if max_frames > -1 and frame_atual >max_frames:
        break
    (H, W) = frame.shape[:2]
    t = time.time()
    if largura_maxima is not None:
        frame = cv2.resize(frame, (video_largura, video_altura))
    cv2.putText(frame, " frame processando em {:.2f} segundos".format(time.time() - t), (20, video_altura - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 0, lineType = cv2.LINE_AA)
    if frame_atual <= frames_show:
        cv2.imshow('Video Processado', frame)
        # cv2.imshow(cv2.resize(frame, (0,0) , fx = 0.75, fy= 0.75))
    frame_atual += 1
print('Terminou!')

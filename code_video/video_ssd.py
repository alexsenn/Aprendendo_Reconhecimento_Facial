import cv2  # OpenCV
import numpy as np
import time

# Carregar o vídeo
arquivo_video = 'Arquivos/videos/video01.mp4'
cap = cv2.VideoCapture(arquivo_video)

# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()

# Leitura do primeiro frame para obter dimensões
conectado, video = cap.read()
if not conectado:
    print("Erro: Não foi possível ler o vídeo.")
    exit()

print("Vídeo conectado:", conectado)

# Verifica as dimensões do vídeo
print("Dimensões do vídeo:", video.shape)
video_largura = video.shape[1]
video_altura = video.shape[0]
print("Largura original:", video_largura, "Altura original:", video_altura)

# Redimensionamento do vídeo
largura_maxima = 900

# Função para calcular nova largura e altura
def redimenciona_video(largura, altura, largura_maxima=600):
    if largura > largura_maxima:
        proporcao = largura / altura
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = largura
        video_altura = altura
    return video_largura, video_altura

if largura_maxima is not None:
    video_largura, video_altura = redimenciona_video(video_largura, video_altura, largura_maxima)
print("Nova largura:", video_largura, "Nova altura:", video_altura)

# Configuração para salvar o vídeo processado
arquivo_resultado = 'resultado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
video_saida = cv2.VideoWriter(arquivo_resultado, fourcc, fps, (video_largura, video_altura))

# Criando o classificador Haar
network = cv2.dnn.readNetFromCaffe("Arquivos/weights/deploy.prototxt.txt", "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel")

# Função para detectar faces usando Haar Cascade
def detecta_face_ssd(net, imagem, show_conf=True, tamanho=300, conf_min = 0.7):
    (h, w) = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (tamanho, tamanho)), 1.0, (tamanho, tamanho), (104.0, 117.0, 123.0))
    net.setInput(blob)
    deteccoes = net.forward()

    face = None
    for i in range(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]
        if confianca > conf_min:
            box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
        
            cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0,), 2)
            if show_conf:
                text_conf = "{:.2f}%".format(confianca * 100)
                cv2.putText(imagem, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    return imagem

# Processamento do vídeo frame por frame
frames_show = 200
frame_atual = 1
max_frames = -1  # -1 para processar todos os frames

while True:
    conectado, frame = cap.read()
    if not conectado:
        print("Fim do vídeo ou erro na leitura.")
        break
    if max_frames > -1 and frame_atual > max_frames:
        print(f"Parou após {max_frames} frames.")
        break

    # Redimensiona o frame, se necessário
    if largura_maxima is not None:
        frame = cv2.resize(frame, (video_largura, video_altura))

    # Medição do tempo de processamento
    t = time.time()
    frame_processado = detecta_face_ssd(network, frame, True, 500)
    tempo_processamento = time.time() - t

    # Adiciona texto com o tempo de processamento
    cv2.putText(frame_processado, "Frame processado em {:.2f} segundos".format(tempo_processamento), 
                (20, video_altura - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1, cv2.LINE_AA)

    # Escreve o frame no vídeo de saída
    video_saida.write(frame_processado)

    # Exibe os primeiros 'frames_show' frames
    if frame_atual <= frames_show:
        cv2.imshow('Video Processado', frame_processado)

    frame_atual += 1

    # Permite sair pressionando 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
video_saida.release()
cv2.destroyAllWindows()
print('Processamento concluído!')

import cv2
import numpy as np
import os

# Função para detectar faces usando SSD e retornar a ROI da face
def detect_face_ssd(network, orig_frame, conf_min=0.5, input_size=300):
    frame = orig_frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (input_size, input_size)), 1.0, 
                                 (input_size, input_size), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = bbox.astype("int")
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w, end_x), min(h, end_y)

            if end_x <= start_x or end_y <= start_y:
                continue

            # Extrai a ROI da face e redimensiona para 60x80 (tamanho usado no treinamento)
            roi = frame[start_y:end_y, start_x:end_x]
            face = cv2.resize(roi, (60, 80))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Desenha retângulo e confiança na imagem
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            text_conf = "{:.2f}%".format(confidence * 100)
            y_text = start_y - 10 if start_y - 10 > 10 else start_y + 20
            cv2.putText(frame, text_conf, (start_x, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return face_gray, frame  # Retorna a face em escala de cinza e o frame processado
    
    return None, frame  # Retorna None se nenhuma face for detectada

# Configurações iniciais
MODEL_PROTO = "Arquivos/weights/deploy.prototxt.txt"
MODEL_WEIGHTS = "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel"
MAX_WIDTH = 1280  # Largura máxima para redimensionar o frame da webcam

# Carrega o modelo SSD
try:
    network = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
except cv2.error as e:
    print(f"Erro ao carregar o modelo SSD: {e}")
    exit()

# Carrega o classificador Eigenfaces
if os.path.exists('eigen_classifier.yml'):
    print("Arquivo 'eigen_classifier.yml' encontrado. Carregando classificador existente...")
    eigen_classifier = cv2.face.EigenFaceRecognizer_create()
    eigen_classifier.read('eigen_classifier.yml')
else:
    print("Erro: Arquivo 'eigen_classifier.yml' não encontrado. Treine o classificador primeiro.")
    exit()

# Inicialização da captura de vídeo
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

# Loop principal para análise em tempo real
while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Redimensiona o frame para um tamanho gerenciável
    if MAX_WIDTH is not None:
        scale = MAX_WIDTH / frame.shape[1]
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Detecta a face no frame
    face, processed_frame = detect_face_ssd(network, frame, conf_min=0.5)

    # Se uma face for detectada, faz a previsão com Eigenfaces
    if face is not None:
        previsao = eigen_classifier.predict(face)
        pred_id = previsao[0]  # ID previsto
        confidence = previsao[1]  # Confiança da previsão (distância)

        # Adiciona o ID previsto no frame
        cv2.putText(processed_frame, f'ID: {pred_id}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'Conf: {confidence:.2f}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Exibe o frame processado
    cv2.imshow('Reconhecimento Facial em Tempo Real', processed_frame)

    # Sai do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cam.release()
cv2.destroyAllWindows()
print("Processamento concluído!")
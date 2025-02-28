import cv2
import numpy as np
from webcam_helper_functions import resize_video

# Configurações iniciais
DETECTOR_TYPE = "ssd"  # Opções: "haarcascade" ou "ssd"
MAX_WIDTH = 1800       # Largura máxima do vídeo (None para manter original)
CONF_MIN = 0.5         # Confiança mínima para SSD

# Função para detectar faces usando Haar Cascade
def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()  # Preserva o frame original
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    return frame

# Função para detectar faces usando SSD
def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=CONF_MIN):
    frame = orig_frame.copy()  # Preserva o frame original
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = bbox.astype("int")

            # Evita coordenadas fora dos limites da imagem
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w, end_x), min(h, end_y)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                y_text = start_y - 10 if start_y - 10 > 10 else start_y + 20  # Ajusta texto se perto da borda
                cv2.putText(frame, text_conf, (start_x, y_text), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Inicialização do detector
if DETECTOR_TYPE == "haarcascade":
    face_detector = cv2.CascadeClassifier('Arquivos/cascades/haarcascade_frontalface_default.xml')
    if face_detector.empty():
        print("Erro: Não foi possível carregar o classificador Haar.")
        exit()
else:  # SSD
    try:
        network = cv2.dnn.readNetFromCaffe("Arquivos/weights/deploy.prototxt.txt", 
                                           "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel")
    except cv2.error as e:
        print(f"Erro ao carregar o modelo SSD: {e}")
        exit()

# Inicialização da captura de vídeo
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

# Loop principal
while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Redimensiona o frame, se necessário
    if MAX_WIDTH is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], MAX_WIDTH)
        frame = cv2.resize(frame, (video_width, video_height))

    # Processa o frame com o detector escolhido
    if DETECTOR_TYPE == "haarcascade":
        processed_frame = detect_face(face_detector, frame)
    else:
        processed_frame = detect_face_ssd(network, frame)

    # Exibe o frame processado
    cv2.imshow("Detecting Faces", processed_frame)

    # Sai do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cam.release()
cv2.destroyAllWindows()
print("Processamento concluído!")
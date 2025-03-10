import cv2
import numpy as np
from PIL import Image
import os

# Tamanho fixo para todas as faces
TAMANHO_FACE = (80, 80)  # Ajuste conforme necessário (ex.: 60x80 ou outro)

def detecta_face(network, path_imagem, conf_min=0.5, input_size=300, margem=0.0):
    """
    Detecta uma face em uma imagem usando SSD e retorna a ROI da face e a imagem processada.
    
    Parâmetros:
    - network: Modelo SSD carregado.
    - path_imagem (str): Caminho da imagem.
    - conf_min (float): Limiar de confiança.
    - input_size (int): Tamanho de entrada do SSD.
    - margem (float): Margem adicional ao redor da face detectada (ex.: 0.2 = 20%). 
    
    Retorna:
    - face: ROI da face redimensionada e normalizada.
    - imagem_colorida: Imagem com retângulo desenhado.
    """
    if not os.path.exists(path_imagem):
        print(f"Erro: O arquivo '{path_imagem}' não foi encontrado.")
        return None, None

    try:
        imagem = Image.open(path_imagem).convert('L')
    except Exception as e:
        print(f"Erro ao abrir a imagem '{path_imagem}' com PIL: {e}")
        return None, None

    imagem = np.array(imagem, dtype='uint8')
    imagem_colorida = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    h, w = imagem_colorida.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(imagem_colorida, (input_size, input_size)), 
                                 1.0, (input_size, input_size), (104.0, 117.0, 123.0))
    network.setInput(blob)
    deteccoes = network.forward()

    face = None
    for i in range(deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]
        if confianca > conf_min:
            bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = bbox.astype('int')

            # Adicionar margem ao redor da face detectada
            largura = end_x - start_x
            altura = end_y - start_y
            margem_x = int(largura * margem)
            margem_y = int(altura * margem)

            start_x = max(0, start_x - margem_x)
            start_y = max(0, start_y - margem_y)
            end_x = min(w, end_x + margem_x)
            end_y = min(h, end_y + margem_y)

            if end_x <= start_x or end_y <= start_y:
                continue

            roi = imagem_colorida[start_y:end_y, start_x:end_x]
            # Redimensionar para o tamanho fixo
            face = cv2.resize(roi, TAMANHO_FACE, interpolation=cv2.INTER_AREA)
            # Converter para escala de cinza e normalizar
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)  # Normalização de histograma para consistência

            # Desenhar retângulo na imagem original
            cv2.rectangle(imagem_colorida, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            text_conf = "{:.2f}%".format(confianca * 100)
            y_text = start_y - 10 if start_y - 10 > 10 else start_y + 20
            cv2.putText(imagem_colorida, text_conf, (start_x, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            break

    return face, imagem_colorida

def get_image_data(network, dataset_path):
    """Cria uma base de dados com IDs e faces detectadas."""
    if not os.path.exists(dataset_path):
        print(f"Erro: O diretório '{dataset_path}' não foi encontrado.")
        return np.array([]), []

    paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
             if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.jpg', '.png'))]
    
    if not paths:
        print(f"Aviso: Nenhum arquivo de imagem encontrado em '{dataset_path}'.")
        return np.array([]), []

    faces = []
    ids = []
    for path in paths:
        face, _ = detecta_face(network, path)  # Não precisamos da imagem aqui
        if face is None:
            print(f"Aviso: Nenhuma face detectada em '{path}'.")
            continue

        filename = os.path.split(path)[1]
        try:
            id_str = filename.split('.')[1]
            id = int(id_str)
        except (IndexError, ValueError) as e:
            print(f"Erro ao extrair ID de '{filename}': {e}")
            continue

        ids.append(id)
        faces.append(face)

    return np.array(ids), faces
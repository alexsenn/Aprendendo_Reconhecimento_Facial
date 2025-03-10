import cv2
import numpy as np
from PIL import Image
import os
from data_base_new import detecta_face
from sklearn.metrics import accuracy_score, confusion_matrix  # Adiciona confusion_matrix
import seaborn as sns  # Importação do seaborn
import matplotlib.pyplot as plt  # Necessário para exibir os gráficos

# Configurações iniciais
MODEL_PROTO = "Arquivos/weights/deploy.prototxt.txt"
MODEL_WEIGHTS = "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel"
CLASSIFIER_PATH = "fisher_classifier.yml"
TEST_PATH = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/image_tests "

# Carrega o modelo SSD
network = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

# Carrega o classificador FisherFace
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"O arquivo '{CLASSIFIER_PATH}' não foi encontrado. Treine o classificador primeiro.")
classificador = cv2.face.FisherFaceRecognizer_create()
classificador.read(CLASSIFIER_PATH)

def avalia_algoritmo(paths, classificador):
    """
    Avalia o desempenho do classificador em um conjunto de imagens de teste.
    
    Parâmetros:
    - paths (list): Lista de caminhos das imagens de teste.
    - classificador: Objeto EigenFaceRecognizer treinado.
    
    Retorna:
    - previsoes (ndarray): Array com IDs previstos.
    - saidas_esperadas (ndarray): Array com IDs esperados.
    """
    previsoes = []
    saidas_esperadas = []
    for path in paths:
        face, imagem = detecta_face(network, path)
        if face is None:
            print(f"Aviso: Nenhuma face detectada em '{path}'. Ignorando...")
            continue
        
        try:
            previsao, conf = classificador.predict(face)
            saida_esperada = int(os.path.split(path)[1].split('.')[1])  # Converte para inteiro
            previsoes.append(previsao)
            saidas_esperadas.append(saida_esperada)
        except cv2.error as e:
            print(f"Erro ao prever a face em '{path}': {e}")
            continue
    
    return np.array(previsoes), np.array(saidas_esperadas)

# Define os caminhos de teste
diretorio_teste = TEST_PATH
paths_teste = [os.path.join(diretorio_teste, f) for f in os.listdir(diretorio_teste) if f.endswith(('.jpg', '.png'))]

# Avalia o algoritmo
previsoes, saidas_esperadas = avalia_algoritmo(paths_teste, classificador)
print("Previsões:", previsoes)
print("Saídas esperadas:", saidas_esperadas)

# Calcula a acurácia
if len(previsoes) > 0 and len(saidas_esperadas) > 0:
    acuracia = accuracy_score(saidas_esperadas, previsoes)
    print(f"Acurácia do algoritmo: {acuracia:.2%}")

    # Calcula a matriz de confusão
    cm = confusion_matrix(saidas_esperadas, previsoes)

    # Visualiza a matriz de confusão com seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsões")
    plt.ylabel("Saídas Esperadas")
    plt.show()
else:
    print("Não há previsões válidas para calcular a acurácia ou matriz de confusão.")
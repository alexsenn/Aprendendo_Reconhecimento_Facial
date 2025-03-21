import cv2  # OpenCV
import numpy as np

# Caminhos dos arquivos do modelo
arquivo_modelo = 'Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel'
arquivo_prototxt = 'Arquivos/weights/deploy.prototxt.txt'

# Carrega o modelo DNN
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)

# Confiança mínima para considerar uma detecção
conf_min = 0.5  

# Carrega a imagem
imagem = cv2.imread('Arquivos/images/people2.jpg')

# Obtém as dimensões da imagem original
(h, w) = imagem.shape[:2]  # Altura e largura da imagem

# Prepara o blob para a rede
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

# Define a entrada da rede e realiza a detecção
network.setInput(blob)
deteccoes = network.forward()

# Loop sobre as detecções
for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]  # Probabilidade da detecção
    if confianca > conf_min:
        # Extrai as coordenadas do retângulo e converte para valores absolutos
        box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype(int)

        # Desenha o retângulo na imagem
        cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Adiciona o texto com a confiança
        text_conf = "{:.2f}%".format(confianca * 100)
        cv2.putText(imagem, text_conf, (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Exibe a imagem com as detecções
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
cv2.destroyAllWindows()  # Fecha a janela
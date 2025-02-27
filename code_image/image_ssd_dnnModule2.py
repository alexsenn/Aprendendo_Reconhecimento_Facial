import cv2
import numpy as np

def detecta_face_ssd(net, path_imagem, tamanho=300, conf_min=0.5):
    """
    Detecta faces em uma imagem usando um modelo SSD pré-treinado.
    
    Args:
        net: Rede neural carregada (cv2.dnn.readNetFromCaffe).
        path_imagem (str): Caminho para a imagem.
        tamanho (int): Tamanho para redimensionar a imagem (padrão: 300).
        conf_min (float): Confiança mínima para aceitar uma detecção (padrão: 0.5).
    
    Returns:
        None: Exibe a imagem com as detecções em uma janela.
    """
    # Carrega a imagem
    imagem = cv2.imread(path_imagem)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem em '{path_imagem}'")
        return

    # Obtém as dimensões da imagem
    (h, w) = imagem.shape[:2]

    # Prepara o blob para a rede
    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (tamanho, tamanho)), 1.0, 
                                 (tamanho, tamanho), (104.0, 117.0, 123.0))

    # Define a entrada da rede e realiza a detecção
    net.setInput(blob)
    deteccoes = net.forward()

    # Loop sobre as detecções
    for i in range(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]

        if confianca > conf_min:
            # Extrai as coordenadas do retângulo
            box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Desenha o retângulo e o texto na imagem
            text_conf = "{:.2f}%".format(confianca * 100)
            cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(imagem, text_conf, (start_x, start_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    # Exibe a imagem com as detecções
    cv2.imshow('Imagem com Detecoes', imagem)
    cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
    cv2.destroyAllWindows()  # Fecha a janela

# Exemplo de uso
if __name__ == "__main__":
    # Caminhos dos arquivos do modelo
    arquivo_modelo = 'Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel'
    arquivo_prototxt = 'Arquivos/weights/deploy.prototxt.txt'

    # Carrega o modelo
    try:
        network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)
    except cv2.error as e:
        print(f"Erro ao carregar o modelo: {e}")
        exit()

    # Chama a função
    detecta_face_ssd(network, 'Arquivos/images/more_people.jpg', 1000)
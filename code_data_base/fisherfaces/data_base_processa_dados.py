import cv2
import numpy as np
from data_base_new import detecta_face, get_image_data
import os

def processar_reconhecimento_facial(model_proto, model_weights, dataset_path, imagem_teste=None, classifier_path=None):
    """
    Processa o reconhecimento facial com FisherFace: cria base de dados, treina ou carrega o classificador e testa.

    Parâmetros:
    - model_proto (str): Caminho do arquivo prototxt do modelo SSD.
    - model_weights (str): Caminho dos pesos do modelo SSD.
    - dataset_path (str): Diretório com as imagens de treinamento.
    - imagem_teste (str, opcional): Caminho da imagem de teste.
    - classifier_path (str): Caminho para salvar/carregar o classificador.

    Retorna:
    - imagem_resultado (ndarray): Imagem de teste com previsões (se fornecida).
    - previsao (tuple): Resultado da previsão (ID previsto, confiança) ou None.
    """
    # Carrega o modelo SSD
    try:
        network = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
    except cv2.error as e:
        print(f"Erro ao carregar o modelo SSD: {e}")
        return None, None
   
    # Verifica se o classificador já existe
    fisher_classifier = cv2.face.FisherFaceRecognizer()
    if os.path.exists(classifier_path):
        print(f"Carregando classificador existente de '{classifier_path}'...")
        fisher_classifier.read(classifier_path)
    else:
        # Cria a base de dados
        ids, faces = get_image_data(network, dataset_path)
        if len(ids) == 0 or not faces:
            print("Nenhuma face válida detectada na base de dados.")
            return None, None

        print(f"Base de dados criada com {len(ids)} faces.")
        print("Treinando novo classificador FisherFace...")
        try:
            fisher_classifier.train(faces, ids)
            fisher_classifier.write(classifier_path)
            print(f"Classificador treinado e salvo em '{classifier_path}'.")
        except cv2.error as e:
            print(f"Erro ao treinar o classificador FisherFace: {e}")
            return None, None

    # Teste com imagem fornecida (se houver)
    if imagem_teste:
        face, imagem = detecta_face(network, imagem_teste)
        if face is None:
            print("Nenhuma face detectada na imagem de teste.")
            return None, None

        # Faz a previsão
        previsao = fisher_classifier.predict(face)
        saida_esperada = os.path.split(imagem_teste)[1].split('.')[1]

        # Adiciona texto na imagem
        cv2.putText(imagem, f'Pred: {previsao[0]}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        cv2.putText(imagem, f'Exp: {saida_esperada}', (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        return imagem, previsao

    return None, None

if __name__ == "__main__":
    # Configurações iniciais
    MODEL_PROTO = "Arquivos/weights/deploy.prototxt.txt"
    MODEL_WEIGHTS = "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel"
    DATASET_PATH = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI"
    IMAGE_TEST = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/image_tests/Pessoa.005.teste2.png"
    # IMAGE_TEST = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/Pessoa.010.J.jpg(8).jpg"
    CLASSIFIER_PATH='fisher_classifier.yml'
    
    # Executa a função
    imagem_resultado, previsao = processar_reconhecimento_facial(MODEL_PROTO, MODEL_WEIGHTS, DATASET_PATH, IMAGE_TEST, CLASSIFIER_PATH)
    
    if imagem_resultado is not None:
        cv2.imshow('Imagem Teste', imagem_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
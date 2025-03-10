import cv2
import numpy as np
from data_base_new import detecta_face, get_image_data
import os

def teste_reconhecimento(imagem_teste, classificador, show_conf=False):
    """Testa o reconhecimento facial em uma imagem e retorna a imagem anotada e a previsão."""
    face, imagem_np = detecta_face(network, imagem_teste)
    if face is None:
        print(f"Nenhuma face detectada em '{imagem_teste}'.")
        return None, None

    try:
        previsao, conf = classificador.predict(face)
        # Ajusta a extração do ID esperado com base no nome do arquivo
        nome_arquivo = os.path.split(imagem_teste)[1]  # Ex.: Pessoa.005.teste2.png
        saida_esperada = int(nome_arquivo.split('.')[1])  # Extrai '005' e converte para inteiro
        
        cv2.putText(imagem_np, 'Pred: ' + str(previsao), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        cv2.putText(imagem_np, 'Exp: ' + str(saida_esperada), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        if show_conf:
            print(f"Confiança da previsão: {conf}")
        
        return imagem_np, previsao
    except cv2.error as e:
        print(f"Erro ao fazer a previsão: {e}")
        return None, None

def ajusta_parametro(classifier_path, dataset_path, valor_parametro):
    """Treina um classificador FisherFace com um parâmetro específico e retorna o classificador."""
    # Cria a base de dados
    ids, faces = get_image_data(network, dataset_path)
    if len(ids) == 0 or not faces:
        print("Nenhuma face válida detectada na base de dados.")
        return None
    
    print(f"Base de dados criada com {len(ids)} faces.")
    print("Treinando novo classificador FisherFace...")
    try:
        fisher_classifier = cv2.face.FisherFaceRecognizer_create(num_components=valor_parametro)
        fisher_classifier.train(faces, ids)
        fisher_classifier.write(classifier_path)
        print(f"Classificador treinado e salvo em '{classifier_path}'.")
        return fisher_classifier
    except cv2.error as e:
        print(f"Erro ao treinar o classificador FisherFace: {e}")
        return None

if __name__ == "__main__":
    # Configurações iniciais
    MODEL_PROTO = "Arquivos/weights/deploy.prototxt.txt"
    MODEL_WEIGHTS = "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel"
    DATASET_PATH = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI"
    IMAGE_TEST = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/image_tests2/Pessoa.005.teste.png"
    CLASSIFIER_PATH = "fisher_classifier.yml"
    
    # Carrega o modelo SSD
    network = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
    
    # Valor do parâmetro de calibragem do FisherFace
    VALOR_PARAMETRO = 40

    # Treina ou carrega o classificador
    classificador = ajusta_parametro(CLASSIFIER_PATH, DATASET_PATH, VALOR_PARAMETRO)
    if classificador is None:
        print("Erro ao treinar o classificador. Encerrando...")
        exit()

    # Testa o reconhecimento
    imagem_np, prediction = teste_reconhecimento(IMAGE_TEST, classificador, True)
    if imagem_np is not None:
        cv2.imshow("Resultado", imagem_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum resultado para exibir.")
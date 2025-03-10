import cv2
import numpy as np
from data_new import detecta_face, get_image_data
import os

# Configurações iniciais
MODEL_PROTO = "Arquivos/weights/deploy.prototxt.txt"
MODEL_WEIGHTS = "Arquivos/weights/res10_300x300_ssd_iter_140000.caffemodel"
DATASET_PATH = "Arquivos/datasets/yalefaces/yalefaces/Fotos_TI"

# Carrega o modelo SSD
try:
    network = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
except cv2.error as e:
    print(f"Erro ao carregar o modelo SSD: {e}")
    exit()

# Executa a criação da base de dados
ids, faces = get_image_data(network, DATASET_PATH)

# Verifica se há dados válidos
if len(ids) > 0 and faces:
    print(f"Base de dados criada com {len(ids)} faces.")
    
    # Verifica se o classificador já existe
    if os.path.exists('eigen_classifier.yml'):
        print("Arquivo 'eigen_classifier.yml' encontrado. Carregando classificador existente...")
        eigen_classifier = cv2.face.EigenFaceRecognizer_create()
        eigen_classifier.read('eigen_classifier.yml')

        # Caminho da imagem de teste
        image_test = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/image_tests/Pessoa.021.teste.jpg'

        # Detecta a face na imagem de teste
        face, imagem = detecta_face(network, image_test)
        if face is None:
            print("Nenhuma face detectada na imagem de teste.")
            exit()

        # Faz a previsão com o classificador Eigenfaces
        previsao = eigen_classifier.predict(face)
        saida_esperada = os.path.split(image_test)[1].split('.')[1]

        # Adiciona texto na imagem original
        cv2.putText(imagem, 'Pred: ' + str(previsao[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        cv2.putText(imagem, 'Exp: ' + str(saida_esperada), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))

        # Redimensiona a imagem para exibição (ex.: 25% do tamanho original)
        # scale_percent = 25  # Reduz para 25% (ajuste conforme necessário)
        # width = int(imagem.shape[1] * scale_percent / 100)
        # height = int(imagem.shape[0] * scale_percent / 100)
        # imagem_resized = cv2.resize(imagem, (width, height), interpolation=cv2.INTER_AREA)

        # Exibe a imagem redimensionada
        cv2.imshow('Imagem Teste', imagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Arquivo 'eigen_classifier.yml' não encontrado. Treinando novo classificador...")
        try:
            eigen_classifier = cv2.face.EigenFaceRecognizer_create()
            eigen_classifier.train(faces, ids)
            eigen_classifier.write('eigen_classifier.yml')
            print("Classificador Eigenfaces treinado e salvo como 'eigen_classifier.yml'.")
        except AttributeError as e:
            print(f"Erro: O módulo 'cv2.face' não está disponível. Instale 'opencv-contrib-python'. Detalhes: {e}")
            exit()
        except cv2.error as e:
            print(f"Erro ao treinar o classificador Eigenfaces: {e}")
            exit()

    # Exibe algumas faces para verificação (sem padding)
    # for i, (id, face) in enumerate(zip(ids, faces)):
    #     if i >= 5:
    #         break
    #     cv2.imshow(f'Face ID {id}', face)
    #     cv2.waitKey(0)  # Exibe uma por vez, pressione uma tecla para a próxima
    # cv2.destroyAllWindows()

    # Informações adicionais
    print(f"Total de IDs: {len(ids)}, Total de faces: {len(faces)}")
    if faces:
        print(f"Dimensões da primeira face: {faces[0].shape}")
else:
    print("Nenhuma face válida detectada na base de dados.")
    exit()
import cv2
import numpy as np
from PIL import Image
import os
import dlib
import pickle

def extrai_descritor(face, imagem_rgb, descritores_faces, extrator_descritor_facial, detector_pontos):
    """
    Extrai o descritor facial de uma face detectada e desenha retângulos/pontos na imagem.

    Args:
        face: Objeto de face detectada pelo dlib.
        imagem_rgb: Imagem em formato RGB (array NumPy).
        descritores_faces: Array de descritores faciais acumulados ou None.
        extrator_descritor_facial: Modelo dlib para extrair descritores.
        detector_pontos: Preditor de pontos faciais do dlib.

    Returns:
        imagem_rgb: Imagem RGB com retângulos e pontos desenhados.
        descritores_faces: Array atualizado com o novo descritor.
    """
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem_rgb, (l, t), (r, b), (0, 0, 255), 2)  # Vermelho em RGB

    pontos = detector_pontos(imagem_rgb, face)
    for ponto in pontos.parts():
        cv2.circle(imagem_rgb, (ponto.x, ponto.y), 2, (0, 255, 0), 1)  # Verde em RGB

    descritor_face = extrator_descritor_facial.compute_face_descriptor(imagem_rgb, pontos)
    descritor_face = np.array([f for f in descritor_face], dtype=np.float64)
    descritor_face = descritor_face[np.newaxis, :]

    if descritores_faces is None:
        descritores_faces = descritor_face
    else:
        descritores_faces = np.concatenate((descritores_faces, descritor_face), axis=0)

    return imagem_rgb, descritores_faces

def carrega_treinamento(path_dataset, detector_face, extrator_descritor_facial, detector_pontos):
    """
    Carrega imagens de treinamento, extrai descritores faciais e cria um índice.

    Args:
        path_dataset: Caminho do diretório com imagens de treinamento.
        detector_face: Detector de faces do dlib.
        extrator_descritor_facial: Extrator de descritores do dlib.
        detector_pontos: Preditor de pontos faciais do dlib.

    Returns:
        descritores_faces: Array NumPy com todos os descritores faciais.
        index: Dicionário mapeando índices para caminhos das imagens.
    """
    index = {}
    idx = 0
    descritores_faces = None

    paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for path in paths:
        print(f"Processando: {path}")
        imagem_bgr = cv2.imread(path)
        if imagem_bgr is None:
            print(f"Erro: Não foi possível carregar {path}")
            continue

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        deteccoes_faces = detector_face(imagem_rgb, 1)
        if len(deteccoes_faces) == 0:
            print(f"Aviso: Nenhuma face detectada em {path}")
            continue

        for face in deteccoes_faces:
            imagem_rgb, descritores_faces = extrai_descritor(
                face, imagem_rgb, descritores_faces, extrator_descritor_facial, detector_pontos
            )
            index[idx] = path
            idx += 1

    print(f"Total de faces processadas: {idx}")
    return descritores_faces, index

if __name__ == "__main__":
    # Configurações iniciais
    detector_face = dlib.get_frontal_face_detector()
    detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')
    extrator_descritor_facial = dlib.face_recognition_model_v1('Arquivos/weights/dlib_face_recognition_resnet_model_v1.dat')
    image_test_path = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/'

    # Arquivos para salvar/carregar
    descritores_file = 'descritores_faces.npy'
    index_file = 'index_faces.pickle'

    # Carrega e processa o dataset de treinamento
    if os.path.exists(descritores_file) and os.path.exists(index_file):
        descritores_faces = np.load(descritores_file)
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        print("Carregado de arquivos existentes.")
    else:
        descritores_faces, index = carrega_treinamento(
            image_test_path, detector_face, extrator_descritor_facial, detector_pontos
        )

    if descritores_faces is not None:
        print(f"Total de descritores: {descritores_faces.shape}")
        print(f"Total de índices: {len(index)}")

        # Salva os descritores e o índice
        np.save(descritores_file, descritores_faces)
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        print(f"Descritores salvos em '{descritores_file}' e índice em '{index_file}'")

        # Carrega os arquivos salvos
        descritores_carregados = np.load(descritores_file)
        with open(index_file, 'rb') as f:
            index_carregado = pickle.load(f)

        print(f"Descritores carregados: {descritores_carregados.shape}")
        print(f"Índices carregados: {len(index_carregado)}")
    else:
        print("Nenhum descritor foi gerado.")

    # Testes de Reconhecimento entre as faces salvas
    # print( np.linalg.norm(descritores_faces[90] - descritores_faces[88]))
    # print(index[90])
    # print(index[89])

    # print( np.linalg.norm(descritores_faces[0] - descritores_faces, axis = 1))      #Possui o primeiro elemento na contagem
    # print( np.linalg.norm(descritores_faces[0] - descritores_faces[1:], axis = 1))  #Retira a imagem de comparacao lista

    print( np.argmin(np.linalg.norm(descritores_faces[90] - descritores_faces[1:], axis = 1)))  
    print( np.linalg.norm(descritores_faces[90] - descritores_faces[1:], axis = 1)[89])  
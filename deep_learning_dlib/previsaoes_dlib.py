import cv2
import numpy as np
from PIL import Image
import os
import dlib
import pickle

def extrai_descritor(face, imagem_rgb, extrator_descritor_facial, detector_pontos, tamanho_alvo=(150, 150)):
    """
    Extrai o descritor facial de uma face detectada, recortando e redimensionando a região da face.
    """
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    h, w = b - t, r - l
    margem = int(max(h, w) * 0.2)
    l, t, r, b = max(0, l - margem), max(0, t - margem), min(imagem_rgb.shape[1], r + margem), min(imagem_rgb.shape[0], b + margem)

    face_recortada = imagem_rgb[t:b, l:r]
    if face_recortada.size == 0:
        print("Aviso: Recorte inválido, pulando face.")
        return None

    proporcao = min(tamanho_alvo[0] / face_recortada.shape[0], tamanho_alvo[1] / face_recortada.shape[1])
    novo_h, novo_w = int(face_recortada.shape[0] * proporcao), int(face_recortada.shape[1] * proporcao)
    face_redimensionada = cv2.resize(face_recortada, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    face_final = np.zeros((tamanho_alvo[0], tamanho_alvo[1], 3), dtype=np.uint8)
    y_offset = (tamanho_alvo[0] - novo_h) // 2
    x_offset = (tamanho_alvo[1] - novo_w) // 2
    face_final[y_offset:y_offset+novo_h, x_offset:x_offset+novo_w] = face_redimensionada

    face_ajustada = dlib.rectangle(left=0, top=0, right=face_final.shape[1], bottom=face_final.shape[0])
    pontos = detector_pontos(face_final, face_ajustada)

    descritor_face = extrator_descritor_facial.compute_face_descriptor(face_final, pontos)
    descritor_face = np.array([f for f in descritor_face], dtype=np.float64)
    descritor_face = descritor_face / np.linalg.norm(descritor_face)  # Normaliza como no treinamento
    return descritor_face

def previsoes_dlib(path_dataset, descritores_faces, index, detector_face, detector_pontos, extrator_descritor_facial, threshold=0.35):
    previsoes = []
    saidas_esperadas = []
    paths = sorted([os.path.join(path_dataset, f) for f in os.listdir(path_dataset) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for path in paths:
        imagem = Image.open(path).convert('RGB')
        imagem_np = np.array(imagem, 'uint8')
        h, w = imagem_np.shape[:2]
        deteccoes_faces = detector_face(imagem_np, 1)

        if len(deteccoes_faces) == 0:
            print(f"Aviso: Nenhuma face detectada em {path}")
            continue

        for face in deteccoes_faces:
            descritor_face = extrai_descritor(face, imagem_np, extrator_descritor_facial, detector_pontos)
            if descritor_face is None:
                continue

            descritor_face = descritor_face[np.newaxis, :]
            distancias = np.linalg.norm(descritor_face - descritores_faces, axis=1)
            min_index = np.argmin(distancias)
            distancia_minima = distancias[min_index]

            # Extrai o nome da pessoa do índice (ajustado para seu dataset)
            if distancia_minima <= threshold:
                nome_arquivo = os.path.split(index[min_index])[1]
                nome_previsao = int(nome_arquivo.split('.')[1].split('(')[0])  # Ex.: Pessoa.001.F(1) -> 001
            else:
                nome_previsao = -1  # Desconhecido

            # Extrai o nome real do arquivo de teste
            nome_arquivo_teste = os.path.split(path)[1]
            nome_real = int(nome_arquivo_teste.split('.')[1].split('(')[0])  # Ex.: Pessoa.001.F(2) -> 001

            previsoes.append(nome_previsao)
            saidas_esperadas.append(nome_real)

            # Desenha na imagem
            cv2.rectangle(imagem_np, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(imagem_np, f'Pred: {nome_previsao}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.putText(imagem_np, f'Exp: {nome_real}', (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.putText(imagem_np, f'{distancia_minima:.3f}', (10, h - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0))

            # Exibe a imagem
            imagem_bgr_display = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2BGR) #converte a imagem para exibis corretamente
            cv2.imshow('Treinamento', imagem_bgr_display)
            # cv2.imshow('Reconhecimento', imagem_np)
            cv2.waitKey(0)  # Pressione qualquer tecla para a próxima imagem
            cv2.destroyAllWindows()

    previsoes = np.array(previsoes)
    saidas_esperadas = np.array(saidas_esperadas)
    return previsoes, saidas_esperadas

if __name__ == "__main__":
    # Inicializa os modelos
    detector_face = dlib.get_frontal_face_detector()
    detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')
    extrator_descritor_facial = dlib.face_recognition_model_v1('Arquivos/weights/dlib_face_recognition_resnet_model_v1.dat')
    image_test_path = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/image_tests'
    descritores_file = 'descritores_faces.npy'
    index_file = 'index_faces.pickle'

    # Carrega os descritores e índices do treinamento
    if os.path.exists(descritores_file) and os.path.exists(index_file):
        descritores_faces = np.load(descritores_file)
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        print("Carregado de arquivos existentes.")
    else:
        raise FileNotFoundError("Arquivos de treinamento não encontrados. Execute o treinamento primeiro.")

    # Normaliza os descritores carregados (como no treinamento)
    descritores_faces = descritores_faces / np.linalg.norm(descritores_faces, axis=1)[:, np.newaxis]

    # Executa as previsões
    previsoes, saidas_esperadas = previsoes_dlib(
        image_test_path, descritores_faces, index, detector_face, detector_pontos, extrator_descritor_facial, threshold=0.35
    )

    # Calcula e exibe a acurácia
    acertos = np.sum(previsoes == saidas_esperadas)
    total = len(previsoes)
    acuracia = acertos / total if total > 0 else 0
    print(f"Acurácia: {acuracia:.2%} ({acertos}/{total})")
    print("Previsões:", previsoes)
    print("Saídas esperadas:", saidas_esperadas)
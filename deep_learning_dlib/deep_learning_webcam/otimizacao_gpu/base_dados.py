import cv2
import numpy as np
import cupy as cp
from PIL import Image
import os
import dlib
import pickle

NUMERO_PARA_NOME = {
    "001": "Meire",
    "002": "Bruna",
    "003": "Felipe",
    "004": "Jean",
    "005": "Vitor",
    "006": "Henrique",
    "007": "Daniel",
    "008": "Bruno",
    "009": "Kae",
    "010": "Matheus",
    "011": "Alex",
    "012": "Heitor",
    "013": "Allan",
    "014": "Lucas",
    "015": "Paulo",
    "016": "Ruan",
    "017": "Alexandre",
    "018": "Fabio",
    "019": "Robson",
    "020": "Vinicius",
    "021": "Gilberto"
}

def extrai_descritor(face, imagem_rgb, descritores_faces, extrator_descritor_facial, detector_pontos, tamanho_alvo=(150, 150)):
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    h, w = b - t, r - l
    margem = int(max(h, w) * 0.2)
    l, t, r, b = max(0, l - margem), max(0, t - margem), min(imagem_rgb.shape[1], r + margem), min(imagem_rgb.shape[0], b + margem)

    face_recortada = imagem_rgb[t:b, l:r]
    if face_recortada.size == 0:
        print("Aviso: Recorte inválido, pulando face.")
        return imagem_rgb, descritores_faces

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(face_recortada)
    proporcao = min(tamanho_alvo[0] / face_recortada.shape[0], tamanho_alvo[1] / face_recortada.shape[1])
    novo_h, novo_w = int(face_recortada.shape[0] * proporcao), int(face_recortada.shape[1] * proporcao)
    gpu_resized = cv2.cuda.resize(gpu_frame, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    face_redimensionada = gpu_resized.download()

    face_final = np.zeros((tamanho_alvo[0], tamanho_alvo[1], 3), dtype=np.uint8)
    y_offset = (tamanho_alvo[0] - novo_h) // 2
    x_offset = (tamanho_alvo[1] - novo_w) // 2
    face_final[y_offset:y_offset+novo_h, x_offset:x_offset+novo_w] = face_redimensionada

    cv2.rectangle(imagem_rgb, (l, t), (r, b), (0, 0, 255), 2)
    pontos_originais = detector_pontos(imagem_rgb, face)
    for ponto in pontos_originais.parts():
        cv2.circle(imagem_rgb, (ponto.x, ponto.y), 2, (0, 255, 0), 1)

    face_ajustada = dlib.rectangle(left=0, top=0, right=face_final.shape[1], bottom=face_final.shape[0])
    pontos = detector_pontos(face_final, face_ajustada)

    descritor_face = extrator_descritor_facial.compute_face_descriptor(face_final, pontos)
    descritor_face = np.array([f for f in descritor_face], dtype=np.float64)
    descritor_face = descritor_face[np.newaxis, :]

    if descritores_faces is None:
        descritores_faces = descritor_face
    else:
        descritores_faces = np.concatenate((descritores_faces, descritor_face), axis=0)

    return imagem_rgb, descritores_faces

def carrega_treinamento(path_dataset, detector_face, extrator_descritor_facial, detector_pontos):
    index = {}
    idx = 0
    descritores_faces = None

    paths = sorted([os.path.join(path_dataset, f) for f in os.listdir(path_dataset) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for path in paths:
        print(f"Processando: {path}")
        imagem_bgr = cv2.imread(path)
        if imagem_bgr is None:
            print(f"Erro: Não foi possível carregar {path}")
            continue

        nome_arquivo = os.path.basename(path)
        partes_nome = nome_arquivo.split('.')
        if len(partes_nome) >= 2:
            numero = partes_nome[1]
            nome = NUMERO_PARA_NOME.get(numero, "Desconhecido")
        else:
            numero = "Desconhecido"
            nome = "Desconhecido"

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        deteccoes_faces = detector_face(imagem_rgb, 1)
        if len(deteccoes_faces) == 0:
            print(f"Aviso: Nenhuma face detectada em {path}")
            continue

        for face in deteccoes_faces:
            imagem_rgb, descritores_faces = extrai_descritor(
                face, imagem_rgb, descritores_faces, extrator_descritor_facial, detector_pontos
            )
            index[idx] = {"path": path, "numero": numero, "nome": nome}
            idx += 1

    print(f"Total de faces processadas: {idx}")
    return descritores_faces, index

if __name__ == "__main__":
    detector_face = dlib.get_frontal_face_detector()
    detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')
    extrator_descritor_facial = dlib.face_recognition_model_v1('Arquivos/weights/dlib_face_recognition_resnet_model_v1.dat')
    image_train_path = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/'
    descritores_file = 'descritores_faces.npy'
    index_file = 'index_faces.pickle'

    if os.path.exists(descritores_file) and os.path.exists(index_file):
        descritores_faces = np.load(descritores_file)
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        print("Carregado de arquivos existentes.")
    else:
        descritores_faces, index = carrega_treinamento(
            image_train_path, detector_face, extrator_descritor_facial, detector_pontos
        )
        np.save(descritores_file, descritores_faces)
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)

    if descritores_faces is not None:
        print(f"Total de descritores: {descritores_faces.shape}")
        print(f"Índices: {len(index)}")
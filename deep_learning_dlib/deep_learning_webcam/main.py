import cv2
import numpy as np
from PIL import Image
import os
import dlib
import pickle
from _cria_base_dados import carrega_treinamento

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

def previsoes_dlib_webcam(descritores_faces, index, detector_face, detector_pontos, extrator_descritor_facial, threshold=0.35):
    # Inicializa a webcam (0 é geralmente a câmera padrão)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return None, None

    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        # Converte o frame de BGR (padrão OpenCV) para RGB (padrão dlib)
        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = imagem_rgb.shape[:2]

        # Detecta faces no frame
        deteccoes_faces = detector_face(imagem_rgb, 1)

        if len(deteccoes_faces) == 0:
            cv2.putText(frame, "Nenhuma face detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in deteccoes_faces:
                descritor_face = extrai_descritor(face, imagem_rgb, extrator_descritor_facial, detector_pontos)
                if descritor_face is None:
                    continue

                descritor_face = descritor_face[np.newaxis, :]
                distancias = np.linalg.norm(descritor_face - descritores_faces, axis=1)
                min_index = np.argmin(distancias)
                distancia_minima = distancias[min_index]

                # Determina a previsão
                if distancia_minima <= threshold:
                    nome_arquivo = os.path.split(index[min_index])[1]
                    nome_previsao = int(nome_arquivo.split('.')[1].split('(')[0])  # Ex.: Pessoa.001.F(1) -> 001
                else:
                    nome_previsao = -1  # Desconhecido

                # Desenha retângulo e informações no frame
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {nome_previsao}', (face.left(), face.top() - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Dist: {distancia_minima:.3f}', (face.left(), face.bottom() + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Exibe o frame processado
        cv2.imshow('Reconhecimento Facial - Webcam', frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a captura e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()
    return None, None  # Como é em tempo real, não retorna previsões/saídas esperadas

if __name__ == "__main__":
    # Inicializa os modelos
    detector_face = dlib.get_frontal_face_detector()
    detector_pontos = dlib.shape_predictor('Arquivos/weights/shape_predictor_68_face_landmarks.dat')
    extrator_descritor_facial = dlib.face_recognition_model_v1('Arquivos/weights/dlib_face_recognition_resnet_model_v1.dat')
    image_train_path = 'Arquivos/datasets/yalefaces/yalefaces/Fotos_TI/'
    descritores_file = 'descritores_faces.npy'
    index_file = 'index_faces.pickle'

    # Carrega ou cria os descritores e índices do treinamento
    if os.path.exists(descritores_file) and os.path.exists(index_file):
        descritores_faces = np.load(descritores_file)
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        print("Carregado de arquivos existentes.")
    else:
        print("Arquivos de treinamento não encontrados. Criando a base de dados...")
        descritores_faces, index = carrega_treinamento(
            image_train_path, detector_face, extrator_descritor_facial, detector_pontos
        )
        if descritores_faces is None or index is None:
            raise ValueError("Falha ao criar a base de dados. Verifique o dataset e os arquivos de pesos.")
        np.save(descritores_file, descritores_faces)
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        print("Base de dados criada e salva com sucesso.")

    # Normaliza os descritores carregados
    descritores_faces = descritores_faces / np.linalg.norm(descritores_faces, axis=1)[:, np.newaxis]

    # Executa o reconhecimento em tempo real com a webcam
    previsoes, saidas_esperadas = previsoes_dlib_webcam(
        descritores_faces, index, detector_face, detector_pontos, extrator_descritor_facial, threshold=0.35
    )
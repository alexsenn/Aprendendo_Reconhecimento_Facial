import cv2
import numpy as np
from data_new import detecta_face, get_image_data  # Verifique se o nome do arquivo está correto

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

# --------------------------------------------------------------------------------

# Verifica e exibe resultados
if len(ids) > 0 and faces:
    print(f"Base de dados criada com {len(ids)} faces.")
    print(f"IDs únicos: {np.unique(ids)}")  # Mostra os IDs únicos disponíveis

    # Escolha o ID que você quer exibir (exemplo: 1)
    target_id = 9  # Altere este valor para o ID desejado

    # Filtra as faces com o ID específico
    matching_faces = [face for id, face in zip(ids, faces) if id == target_id]

    if matching_faces:
        print(f"Encontradas {len(matching_faces)} faces com ID {target_id}.")
        for i, face in enumerate(matching_faces):
            
            # Exibe a face com o ID no título
            cv2.imshow(f'Face ID {target_id} - Instance {i+1}', face)
    else:
        print(f"Nenhuma face encontrada com ID {target_id}.")

    # Aguarda uma tecla e fecha as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()


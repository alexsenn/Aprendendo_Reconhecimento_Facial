import cv2
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# print(torch.__version__)  # Verifique a versão do PyTorch
print(torch.cuda.is_available())  # Verifique se o CUDA está disponível
print(torch.version.cuda)  # Verifique a versão do CUDA que o PyTorch está usando


print(cv2.__version__)  # Mostra a versão atual
print(cv2.cuda.getCudaEnabledDeviceCount())  # Deve retornar 0 se não houver suporte"""""
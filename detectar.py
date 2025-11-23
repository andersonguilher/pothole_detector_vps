import cv2
from ultralytics import YOLO
import os
import requests

# --- CONFIGURAÇÃO ---
modelo_arquivo = "buracos3.pt"
imagem_teste = "teste_buraco.jpg"
# Link de uma imagem de rua com buracos
url_imagem = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Pothole.jpg/800px-Pothole.jpg"

# 1. Verificar Modelo
if not os.path.exists(modelo_arquivo):
    print(f"ERRO: {modelo_arquivo} não encontrado.")
    exit()

# 2. Baixar Imagem de Teste (se não existir)
if not os.path.exists(imagem_teste):
    print("Baixando imagem de teste da internet...")
    try:
        img_data = requests.get(url_imagem).content
        with open(imagem_teste, 'wb') as handler:
            handler.write(img_data)
        print("Imagem baixada com sucesso!")
    except Exception as e:
        print(f"Erro ao baixar imagem: {e}")
        exit()

# 3. Carregar Modelo
print(f"Carregando {modelo_arquivo}...")
model = YOLO(modelo_arquivo)

# 4. Rodar Detecção na IMAGEM
print(f"Analisando {imagem_teste}...")

# save=True vai salvar uma cópia da imagem com os quadrados desenhados na pasta 'runs/detect/predict'
# show=True tenta abrir uma janela (pode falhar se não tiver interface gráfica, mas vamos tentar)
results = model.predict(source=imagem_teste, save=True, show=True, conf=0.25)

print("-" * 30)
for result in results:
    # Conta quantos objetos (buracos) foram achados
    qtd = len(result.boxes)
    if qtd > 0:
        print(f"✅ SUCESSO! Foram detectados {qtd} buracos na imagem.")
        print(f"Resultado salvo na pasta: {result.save_dir}")
    else:
        print("Nenhum buraco detectado (tente diminuir o conf=0.25).")

print("-" * 30)
print("Pressione qualquer tecla na janela da imagem para fechar (ou feche manualmente).")

# Mantém a janela aberta até apertar uma tecla
cv2.waitKey(0)
cv2.destroyAllWindows()
# Salve como detectar_pro_sahi.py
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_slicing_prediction
from sahi.utils.cv import visualize_object_predictions
import numpy as np

# --- CONFIGURAÇÃO ---
# Use o modelo MEDIUM (buracos_pro.pt) para melhores resultados com SAHI
modelo_arquivo = "buracos_pro.pt" 
pasta_saida = "resultados_sahi"

if not os.path.exists(modelo_arquivo):
    print(f"ERRO: '{modelo_arquivo}' não encontrado.")
    exit()

os.makedirs(pasta_saida, exist_ok=True)

# 1. Configurar o Modelo para o SAHI
print("Carregando modelo com SAHI (Inference Slicing)...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=modelo_arquivo,
    confidence_threshold=0.15, # Confiança mais baixa para pegar os difíceis
    device="cpu" # Mude para "cuda" se tiver placa de vídeo NVIDIA configurada
)

# 2. Selecionar Imagem
root = tk.Tk()
root.withdraw() 
caminho_arquivo = filedialog.askopenfilename(
    title="Selecione a foto do asfalto (Alta Resolução)",
    filetypes=[("Imagens", "*.jpg *.jpeg *.png")]
)
root.destroy()

if not caminho_arquivo:
    exit()

print(f"Processando com fatiamento: {caminho_arquivo}")
print("Isso pode demorar um pouco mais que o normal...")

# 3. A MÁGICA DO SLICING (Fatiamento)
result = get_slicing_prediction(
    caminho_arquivo,
    detection_model,
    slice_height=640, # Tamanho de cada "azulejo"
    slice_width=640,
    overlap_height_ratio=0.2, # Quanto os azulejos se sobrepõem (20%)
    overlap_width_ratio=0.2,
    postprocess_type="NMS",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5
)

print("-" * 30)
print(f"✅ Detecções encontradas: {len(result.object_prediction_list)}")

# 4. Visualizar e Mudar o Nome Manualmente (Gambiarra necessária com SAHI)
# O SAHI não facilita mudar o nome da classe facilmente na visualização padrão.
# Vamos desenhar nós mesmos para ter controle total.

image_cv = cv2.imread(caminho_arquivo)

for prediction in result.object_prediction_list:
    # Pega a caixa delimitadora (bbox)
    bbox = prediction.bbox.to_xyxy()
    x1, y1, x2, y2 = map(int, bbox)
    
    score = prediction.score.value
    
    # --- DESENHO MANUAL ---
    # Cor Azul (BGR)
    color = (255, 0, 0) 
    # Desenha o retângulo
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
    
    # Cria o texto "Buraco 0.XX"
    label = f"Buraco {score:.2f}"
    
    # Desenha o fundo do texto para ficar legível
    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    cv2.rectangle(image_cv, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, -1)
    # Desenha o texto branco por cima
    cv2.putText(image_cv, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


# 5. Salvar e Mostrar
nome_original = os.path.basename(caminho_arquivo)
nome_saida = f"sahi_{nome_original}"
caminho_saida_completo = os.path.join(pasta_saida, nome_saida)

cv2.imwrite(caminho_saida_completo, image_cv)
print(f"✅ Imagem salva em: {os.path.abspath(caminho_saida_completo)}")
print("-" * 30)

# Redimensiona para mostrar na tela se a imagem for gigante
scale_percent = 50 # Mostra com 50% do tamanho
width = int(image_cv.shape[1] * scale_percent / 100)
height = int(image_cv.shape[0] * scale_percent / 100)
dim = (width, height)
resized_show = cv2.resize(image_cv, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Resultado SAHI (Slicing)", resized_show)
print("Pressione qualquer tecla na janela para fechar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
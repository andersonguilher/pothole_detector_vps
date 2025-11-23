import cv2
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

# --- CONFIGURAÇÃO ---
modelo_arquivo = "buracos_pro.pt" 
pasta_saida = "resultados_yolo"

# --- MUDANÇA DE COR AQUI (B, G, R) ---
# Verde Fluorescente = (0, 255, 0)
COR_BORDA = (0, 255, 0)         
COR_PREENCHIMENTO = (0, 255, 0) 
OPACIDADE = 0.4                 # Transparência (40% de tinta)

# 1. Carregar Modelo
if not os.path.exists(modelo_arquivo):
    print(f"ERRO: '{modelo_arquivo}' não encontrado.")
    exit()

print(f"Carregando {modelo_arquivo}...")
model = YOLO(modelo_arquivo)

os.makedirs(pasta_saida, exist_ok=True)

# 2. Selecionar Arquivo
print("Abrindo janela de seleção... (Verifique a barra de tarefas se não aparecer)")
root = tk.Tk()
root.withdraw() 
caminho_arquivo = filedialog.askopenfilename(title="Selecione a foto")
root.destroy()

if not caminho_arquivo:
    print("Nenhuma foto selecionada.")
    exit()

print(f"Processando: {os.path.basename(caminho_arquivo)}")

# 3. Rodar a Detecção
# Usando conf=0.12 e iou=0.5 (configuração que funcionou bem na última foto)
results = model.predict(source=caminho_arquivo, save=False, show=False, conf=0.12, iou=0.5)
result = results[0] 

# Carrega a imagem original
img_original = cv2.imread(caminho_arquivo)
h_img, w_img = img_original.shape[:2]
margem_seguranca = 15 

# Cria o overlay para a tinta
overlay = img_original.copy()

deteccoes_validas = 0

# 4. Desenhar Resultados
if result.masks is not None:
    masks = result.masks.xy
    boxes = result.boxes
    
    for i, mask in enumerate(masks):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        
        # Filtro de Borda (Mantido igual)
        tocou_borda = (x1 < margem_seguranca) or (y1 < margem_seguranca) or \
                      (x2 > w_img - margem_seguranca) or (y2 > h_img - margem_seguranca)
        
        if tocou_borda:
            continue 
        
        # Filtro de Área (< 500px ignora)
        area = (x2 - x1) * (y2 - y1)
        if area < 500:
            continue

        deteccoes_validas += 1

        # A) Desenha a Máscara Verde
        pontos_poly = np.array(mask, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pontos_poly], COR_PREENCHIMENTO)
        
        # B) Desenha o Retângulo e Texto Verde
        cv2.rectangle(img_original, (int(x1), int(y1)), (int(x2), int(y2)), COR_BORDA, 2)
        
        texto = f"Buraco {conf:.2f}"
        t_size = cv2.getTextSize(texto, 0, 0.6, 1)[0]
        # Fundo do texto (Verde)
        cv2.rectangle(img_original, (int(x1), int(y1) - t_size[1] - 5), (int(x1) + t_size[0], int(y1)), COR_BORDA, -1)
        # Texto em Preto (para contrastar com o verde neon)
        cv2.putText(img_original, texto, (int(x1), int(y1) - 5), 0, 0.6, (0, 0, 0), 1)

# 5. Mistura Final
img_final = cv2.addWeighted(overlay, OPACIDADE, img_original, 1 - OPACIDADE, 0)

# Salvar
nome_saida = f"{os.path.splitext(os.path.basename(caminho_arquivo))[0]}_verde.jpg"
caminho_final = os.path.join(pasta_saida, nome_saida)
cv2.imwrite(caminho_final, img_final)

print("-" * 30)
print(f"✅ FINALIZADO! {deteccoes_validas} buraco(s) validado(s).")
print(f"Imagem salva em: {caminho_final}")

# Mostrar
if h_img > 800:
    scale = 800 / h_img
    dim = (int(w_img * scale), 800)
    img_show = cv2.resize(img_final, dim)
else:
    img_show = img_final

cv2.imshow("ConservApp - Verde", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configurações do Modelo e Cores
modelo_arquivo = "buracos_pro.pt"
try:
    model = YOLO(modelo_arquivo)
except Exception as e:
    logging.error(f"Erro ao carregar o modelo YOLO: {e}")

COR_VERDE = (0, 255, 0)
COR_TEXTO = (0, 0, 0) # Preto

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar")
async def detectar_buracos(file: UploadFile = File(...)):
    try:
        # 1. Ler imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Configurações
        h_img, w_img = img_original.shape[:2]
        overlay = img_original.copy()
        margem_seguranca = 15
        buracos_count = 0

        # 3. Rodar YOLO
        results = model.predict(source=img_original, save=False, conf=0.12, iou=0.5)
        result = results[0]

        # 4. Desenhar
        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Filtros
                tocou_borda = (x1 < margem_seguranca) or (y1 < margem_seguranca) or \
                              (x2 > w_img - margem_seguranca) or (y2 > h_img - margem_seguranca)
                area = (x2 - x1) * (y2 - y1)
                
                if tocou_borda or area < 500:
                    continue

                buracos_count += 1

                escala_fonte = 1.5
                espessura = 2
                fonte = cv2.FONT_HERSHEY_SIMPLEX

                pontos_poly = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pontos_poly], COR_VERDE)
                
                cv2.rectangle(img_original, (int(x1), int(y1)), (int(x2), int(y2)), COR_VERDE, 3) 
                
                texto = f"Buraco {conf:.2f}"
                
                (largura_texto, altura_texto), baseline = cv2.getTextSize(texto, fonte, escala_fonte, espessura)
                
                cv2.rectangle(img_original, 
                              (int(x1), int(y1) - altura_texto - 10), 
                              (int(x1) + largura_texto, int(y1)), 
                              COR_VERDE, -1)
                
                cv2.putText(img_original, texto, 
                            (int(x1), int(y1) - 5), 
                            fonte, escala_fonte, COR_TEXTO, espessura)

        # 5. Mistura camadas
        img_final = cv2.addWeighted(overlay, 0.4, img_original, 0.6, 0)

        # 6. Contador no topo (REMOVIDO: Linha que adicionava o texto em vermelho)
        # cv2.putText(img_final, f"Detectados: {buracos_count}", (30, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        _, encoded_img = cv2.imencode('.jpg', img_final)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        logging.error(f"Erro fatal durante a detecção: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Erro interno: {str(e)}"})
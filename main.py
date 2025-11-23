from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import io
import logging
import base64

# Configurar logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Montar a pasta 'templates' e 'static' (necessário para o proxy reverso)
app.mount("/templates", StaticFiles(directory="templates"), name="templates_static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurações do Modelo e Cores
modelo_arquivo = "buracos_pro.pt"
try:
    model = YOLO(modelo_arquivo)
except Exception as e:
    logging.error(f"Erro ao carregar o modelo YOLO: {e}")

COR_VERDE = (0, 255, 0)
COR_TEXTO = (0, 0, 0) # Preto

templates = Jinja2Templates(directory="templates")

# --- PARÂMETROS DE CÁLCULO E CONVERSÃO ---
PIXEL_TO_CM2 = 0.04 
DENSIDADE_CBUQ_TM3 = 2.4 # t/m³
ESPESSURA_CM = 5 # cm

def calcular_area_cm2(area_pixel: float) -> float:
    """Converte área de pixels para cm² usando o fator de calibração."""
    return area_pixel * PIXEL_TO_CM2

def calcular_cbuq_kg(area_base_cm2: float) -> float:
    """Calcula a massa de CBUQ necessária em kg, usando a área base fornecida."""
    area_base_m2 = area_base_cm2 / 10000.0
    espessura_m = ESPESSURA_CM / 100.0
    volume_m3 = area_base_m2 * espessura_m
    massa_kg = volume_m3 * DENSIDADE_CBUQ_TM3 * 1000 # t/m³ * 1000 = kg/m³
    return massa_kg

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
        
        relatorio = []
        buracos_count = 0
        total_dano_cm2 = 0
        total_cbuq_kg = 0

        # 3. Rodar YOLO
        # Limiar de confiança mantido em 0.22
        results = model.predict(source=img_original, save=False, conf=0.22, iou=0.5)
        result = results[0]

        # 4. Desenhar e Coletar Dados
        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Filtros baseados no selecionar_foto.py
                tocou_borda = (x1 < margem_seguranca) or (y1 < margem_seguranca) or \
                              (x2 > w_img - margem_seguranca) or (y2 > h_img - margem_seguranca)
                
                # Cálculo da área de pixel corrigido (largura * altura)
                area_pixel = (x2 - x1) * (y2 - y1) 
                
                if tocou_borda or area_pixel < 500:
                    continue

                buracos_count += 1

                # --- CÁLCULO DAS MÉTRICAS ---
                area_dano_cm2 = calcular_area_cm2(area_pixel)
                massa_cbuq_kg = calcular_cbuq_kg(area_dano_cm2)
                
                # Novas unidades
                area_dano_m2 = area_dano_cm2 / 10000.0   # cm² para m²
                massa_cbuq_t = massa_cbuq_kg / 1000.0    # kg para toneladas
                
                total_dano_cm2 += area_dano_cm2
                total_cbuq_kg += massa_cbuq_kg
                
                relatorio.append({
                    "id": f"#{buracos_count}",
                    "dano_cm2": f"{area_dano_cm2:.2f}",
                    "dano_m2": f"{area_dano_m2:.4f}",
                    "cbuq_kg": f"{massa_cbuq_kg:.2f}",
                    "cbuq_t": f"{massa_cbuq_t:.4f}",
                })

                # --- DESENHO NA IMAGEM ---
                escala_fonte = 1.5
                espessura = 2
                fonte = cv2.FONT_HERSHEY_SIMPLEX

                # 1. Desenha o overlay da máscara (preenchimento verde)
                pontos_poly = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pontos_poly], COR_VERDE)
                
                # 2. Desenha o retângulo original (Dano Real - BBox - Verde)
                cv2.rectangle(img_original, (int(x1), int(y1)), (int(x2), int(y2)), COR_VERDE, 3) 
                
                # 3. Adiciona o texto de confiança (Buraco)
                texto = f"Buraco {buracos_count} ({conf:.2f})"
                
                (largura_texto, altura_texto), baseline = cv2.getTextSize(texto, fonte, escala_fonte * 0.4, espessura)
                
                cv2.rectangle(img_original, 
                              (int(x1), int(y1) - altura_texto - 10), 
                              (int(x1) + largura_texto, int(y1)), 
                              COR_VERDE, -1)
                
                cv2.putText(img_original, texto, 
                            (int(x1), int(y1) - 5), 
                            fonte, escala_fonte * 0.4, COR_TEXTO, 1)

        # 5. Mistura camadas
        img_final = cv2.addWeighted(overlay, 0.4, img_original, 0.6, 0)
        
        # 6. Codifica a imagem final para Base64
        _, encoded_img = cv2.imencode('.jpg', img_final)
        encoded_img_b64 = base64.b64encode(encoded_img).decode('utf-8')
        
        # 7. Calcula totais nas novas unidades
        total_dano_m2 = total_dano_cm2 / 10000.0
        total_cbuq_t = total_cbuq_kg / 1000.0

        # 8. Retorna a imagem e o relatório em JSON
        return JSONResponse(content={
            "image_b64": encoded_img_b64,
            "report_data": relatorio,
            "total_dano_m2": f"{total_dano_m2:.4f}",
            "total_cbuq_t": f"{total_cbuq_t:.4f}",
            "count": buracos_count
        })

    except Exception as e:
        logging.error(f"Erro fatal durante a detecção: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Erro interno: {str(e)}"})
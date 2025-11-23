from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
    """Calcula a massa de CBUQ necessária em kg, usando a área base fornecida (DANO)."""
    area_base_m2 = area_base_cm2 / 10000.0
    espessura_m = ESPESSURA_CM / 100.0
    volume_m3 = area_base_m2 * espessura_m
    massa_kg = volume_m3 * DENSIDADE_CBUQ_TM3 * 1000 # t/m³ * 1000 = kg/m³
    return massa_kg

def processar_arquivo_unico(file_data: bytes, file_name: str):
    """Processa um único arquivo e retorna seus resultados e totais."""
    nparr = np.frombuffer(file_data, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_original is None:
        logging.error(f"Não foi possível decodificar a imagem: {file_name}")
        return None

    h_img, w_img = img_original.shape[:2]
    overlay = img_original.copy()
    margem_seguranca = 15
    
    relatorio = []
    buracos_count = 0
    total_dano_cm2 = 0
    total_cbuq_kg = 0
    
    # Rodar YOLO: Limiar de confiança mantido em 0.22
    results = model.predict(source=img_original, save=False, conf=0.22, iou=0.5)
    result = results[0]

    if result.masks is not None:
        for i, mask in enumerate(result.masks.xy):
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Filtros de borda e área mínima (500 pixels quadrados)
            tocou_borda = (x1 < margem_seguranca) or (y1 < margem_seguranca) or \
                          (x2 > w_img - margem_seguranca) or (y2 > h_img - margem_seguranca)
            area_pixel = (x2 - x1) * (y2 - y1) 
            
            if tocou_borda or area_pixel < 500:
                continue

            buracos_count += 1

            # --- CÁLCULO DAS MÉTRICAS ---
            area_dano_cm2 = calcular_area_cm2(area_pixel)
            massa_cbuq_kg = calcular_cbuq_kg(area_dano_cm2)
            
            total_dano_cm2 += area_dano_cm2
            total_cbuq_kg += massa_cbuq_kg
            
            # Novas unidades
            area_dano_m2 = area_dano_cm2 / 10000.0   # cm² para m²
            massa_cbuq_t = massa_cbuq_kg / 1000.0    # kg para toneladas
            
            relatorio.append({
                "id": f"#{buracos_count}",
                "dano_m2": f"{area_dano_m2:.4f}",
                "cbuq_t": f"{massa_cbuq_t:.4f}",
            })

            # --- DESENHO NA IMAGEM ---
            escala_fonte = 1.5
            espessura = 2
            fonte = cv2.FONT_HERSHEY_SIMPLEX

            pontos_poly = np.array(mask, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pontos_poly], COR_VERDE)
            cv2.rectangle(img_original, (int(x1), int(y1)), (int(x2), int(y2)), COR_VERDE, 3) 
            
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

    # 7. Prepara os totais do arquivo
    total_dano_m2_file = total_dano_cm2 / 10000.0
    total_cbuq_t_file = total_cbuq_kg / 1000.0

    return {
        "file_name": file_name,
        "image_b64": encoded_img_b64,
        "report_data": relatorio,
        "total_dano_m2_file": total_dano_m2_file,
        "total_cbuq_t_file": total_cbuq_t_file,
        "count": buracos_count
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# NOVO ENDPOINT DE LOTE
@app.post("/detectar")
async def detectar_buracos_batch(files: list[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"message": "Nenhum arquivo enviado."})

    batch_results = []
    grand_total_dano_m2 = 0.0
    grand_total_cbuq_t = 0.0
    grand_total_count = 0
    
    for file in files:
        try:
            file_data = await file.read()
            # Processa o arquivo individualmente
            result = processar_arquivo_unico(file_data, file.filename)
            
            if result:
                batch_results.append(result)
                # Acumula os totais gerais
                grand_total_dano_m2 += result['total_dano_m2_file']
                grand_total_cbuq_t += result['total_cbuq_t_file']
                grand_total_count += result['count']
                
        except Exception as e:
            logging.error(f"Erro ao processar o arquivo {file.filename}: {e}", exc_info=True)
            batch_results.append({
                "file_name": file.filename,
                "error": str(e)
            })

    # Retorna o resultado do lote
    return JSONResponse(content={
        "batch_results": batch_results,
        "grand_totals": {
            "total_dano_m2": f"{grand_total_dano_m2:.4f}",
            "total_cbuq_t": f"{grand_total_cbuq_t:.4f}",
            "total_count": grand_total_count
        }
    })
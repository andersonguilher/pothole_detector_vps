from fastapi import FastAPI, File, UploadFile, Request, Form
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
try:
    models = {
        "box": YOLO("best.pt"), # Modelo de detecção de objetos
        "mask": YOLO("buracos_pro.pt") # Modelo de segmentação
    }
except Exception as e:
    logging.error(f"Erro ao carregar os modelos YOLO: {e}")

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

def processar_arquivo_unico(file_data: bytes, file_name: str, confidence_threshold: float, detection_mode: str):
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
    
    # Seleciona o modelo com base no modo
    model = models.get(detection_mode, models["box"])

    # Rodar YOLO
    results = model.predict(source=img_original, save=False, conf=confidence_threshold, iou=0.5, verbose=False)
    result = results[0]

    # Lógica para MÁSCARA DE SEGMENTAÇÃO
    if detection_mode == 'mask' and result.masks is not None:
        for i, mask in enumerate(result.masks):
            # Obter contorno da máscara
            contour = mask.xy[0].astype(np.int32)
            
            # Calcular área em pixels da máscara
            area_pixel = cv2.contourArea(contour)
            if area_pixel < 500: # Filtro de área mínima
                continue

            # Obter confiança do box correspondente
            conf = float(result.boxes[i].conf[0])

            buracos_count += 1
            area_dano_cm2 = calcular_area_cm2(area_pixel)
            massa_cbuq_kg = calcular_cbuq_kg(area_dano_cm2)
            total_dano_cm2 += area_dano_cm2
            total_cbuq_kg += massa_cbuq_kg

            # Desenhar a máscara preenchida
            cv2.fillPoly(overlay, [contour], COR_VERDE)

            # Adicionar etiqueta à máscara (desenhada na imagem original para não ficar transparente)
            x, y, w, h = cv2.boundingRect(contour)
            texto = f"Buraco {buracos_count} ({conf:.2f})"
            (w_texto, h_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Posição do texto acima da máscara
            pos_y_texto = y - 5
            cv2.rectangle(img_original, (x, pos_y_texto - h_texto - 5), (x + w_texto, pos_y_texto + 5), COR_VERDE, -1)
            cv2.putText(img_original, texto, (x, pos_y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO, 1)

            # Adicionar ao relatório
            relatorio.append({
                "id": f"#{buracos_count}",
                "dano_m2": f"{(area_dano_cm2 / 10000.0):.4f}",
                "cbuq_t": f"{(massa_cbuq_kg / 1000.0):.4f}",
                "confidence": f"{conf:.2f}"
            })

    # Lógica para RETÂNGULO (BOX)
    elif detection_mode == 'box' and result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # Filtros
            tocou_borda = (x1 < margem_seguranca) or (y1 < margem_seguranca) or \
                          (x2 > w_img - margem_seguranca) or (y2 > h_img - margem_seguranca)
            area_pixel = (x2 - x1) * (y2 - y1)
            if tocou_borda or area_pixel < 500:
                continue

            buracos_count += 1
            area_dano_cm2 = calcular_area_cm2(area_pixel)
            massa_cbuq_kg = calcular_cbuq_kg(area_dano_cm2)
            total_dano_cm2 += area_dano_cm2
            total_cbuq_kg += massa_cbuq_kg

            # Desenhar retângulo e texto
            cv2.rectangle(img_original, (x1, y1), (x2, y2), COR_VERDE, 3)
            texto = f"Buraco {buracos_count} ({conf:.2f})"
            (w_texto, h_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_original, (x1, y1 - h_texto - 10), (x1 + w_texto, y1), COR_VERDE, -1)
            cv2.putText(img_original, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO, 1)

            # Adicionar ao relatório
            relatorio.append({
                "id": f"#{buracos_count}",
                "dano_m2": f"{(area_dano_cm2 / 10000.0):.4f}",
                "cbuq_t": f"{(massa_cbuq_kg / 1000.0):.4f}",
                "confidence": f"{conf:.2f}"
            })

    # 5. Mistura camadas
    img_final = cv2.addWeighted(overlay, 0.3, img_original, 0.7, 0)
    
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
async def detectar_buracos_batch(
    files: list[UploadFile] = File(...),
    confidence: float = Form(0.22), # Recebe o valor de confiança do formulário
    detection_mode: str = Form("box") # Recebe o modo de detecção: 'box' ou 'mask'
):
    if not files:
        return JSONResponse(status_code=400, content={"message": "Nenhum arquivo enviado."})

    batch_results = []
    grand_total_dano_m2 = 0.0
    grand_total_cbuq_t = 0.0
    grand_total_count = 0
    
    for file in files:
        try:
            file_data = await file.read()
            # Processa o arquivo individualmente, passando a confiança e o modo
            result = processar_arquivo_unico(file_data, file.filename, confidence, detection_mode)
            
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
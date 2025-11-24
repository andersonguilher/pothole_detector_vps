from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates_static")

# --- CARREGAMENTO DOS MODELOS ---
models = {}
print("-" * 30)
try: models["box"] = YOLO("buracos_box.pt"); print("✅ buracos_box.pt (Buracos) carregado.")
except: print("❌ Erro ao carregar buracos_box.pt")

try: models["mask"] = YOLO("buracos_seg.pt"); print("✅ buracos_seg.pt (Mask) carregado.")
except: print("⚠️ buracos_seg.pt não encontrado.")

try: models["bueiro"] = YOLO("bueiros_det.pt"); print("✅ bueiros_det.pt (Bueiros) carregado.")
except: print("❌ bueiros_det.pt não encontrado.")
print("-" * 30)

# --- CONFIGURAÇÕES GERAIS ---
# Resolução de entrada da IA (Padrão YOLO é 640. 1280 melhora objetos pequenos/distantes)
RESOLUCAO_IA = 736

TRADUCAO_BUEIRO = {'good': 'Bom', 'broke': 'Quebrado', 'uncover': 'Aberto'}
CORES_BUEIRO = {
    'Bom': (0, 255, 0),      # Verde
    'Quebrado': (0, 0, 255), # Vermelho
    'Aberto': (0, 0, 255),   # Vermelho
    'Sem Tampa': (0, 0, 255) # Vermelho
}
COR_BURACO = (0, 255, 255)   # Amarelo/Ciano

templates = Jinja2Templates(directory="templates")

# Configuração Visual
FONT_SCALE = 1.2
THICKNESS = 3
PADDING = 10

# Física
PIXEL_TO_CM2 = 0.04 
DENSIDADE_CBUQ_TM3 = 2.4 
ESPESSURA_CM = 5 

# --- FUNÇÕES ---
def calcular_area_cm2(area_pixel): return area_pixel * PIXEL_TO_CM2
def calcular_cbuq_kg(area_cm2): return (area_cm2 / 10000.0 * (ESPESSURA_CM / 100.0)) * DENSIDADE_CBUQ_TM3 * 1000

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def desenhar_etiqueta(img, texto, x, y, cor_fundo, cor_texto=(0,0,0), posicao='topo'):
    (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
    h_img, _, _ = img.shape
    
    if posicao == 'topo':
        y_bg_top = max(0, y - h_txt - 20)
        y_bg_bottom = max(h_txt + 20, y)
        y_txt = max(h_txt + 12, y - 8)
    else:
        y_bg_top = min(h_img - h_txt - 20, y)
        y_bg_bottom = min(h_img, y + h_txt + 20)
        y_txt = min(h_img - 8, y + h_txt + 12)

    cv2.rectangle(img, (x, y_bg_top), (x + w_txt + PADDING, y_bg_bottom), cor_fundo, -1)
    cv2.putText(img, texto, (x + 5, y_txt), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, cor_texto, THICKNESS)

def processar_arquivo_unico(file_data, file_name, conf_buraco, conf_bueiro, detection_mode):
    nparr = np.frombuffer(file_data, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None: return None

    overlay = img_original.copy()
    relatorio = []
    
    raw_bueiros = [] 
    raw_buracos = []

    # 1. DETECÇÃO DE BUEIROS (ALTA RESOLUÇÃO)
    if "bueiro" in models:
        res_b = models["bueiro"].predict(
            source=img_original, 
            save=False, 
            conf=conf_bueiro, 
            iou=0.5, 
            verbose=False,
            imgsz=RESOLUCAO_IA  # <--- APLICA RESOLUÇÃO AQUI
        )
        for box in res_b[0].boxes:
            cls_name = res_b[0].names[int(box.cls[0])]
            label_pt = TRADUCAO_BUEIRO.get(cls_name, cls_name)
            
            # Filtro de rigor para bueiro 'Bom'
            if label_pt == 'Bom' and float(box.conf[0]) < 0.50: continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            raw_bueiros.append({'box': [x1,y1,x2,y2], 'label': label_pt, 'conf': float(box.conf[0])})

    # 2. DETECÇÃO DE BURACOS (ALTA RESOLUÇÃO)
    model_buraco = models.get("mask" if detection_mode == 'mask' else "box")
    if model_buraco:
        res_p = model_buraco.predict(
            source=img_original, 
            save=False, 
            conf=conf_buraco, 
            iou=0.4, 
            verbose=False,
            imgsz=RESOLUCAO_IA  # <--- APLICA RESOLUÇÃO AQUI TAMBÉM
        )
        r = res_p[0]
        
        if detection_mode == 'mask' and r.masks:
            for i, mask in enumerate(r.masks):
                cnt = mask.xy[0].astype(np.int32)
                if cv2.contourArea(cnt) < 500: continue
                x,y,w,h = cv2.boundingRect(cnt)
                area = calcular_area_cm2(cv2.contourArea(cnt))
                raw_buracos.append({'box': [x,y,x+w,y+h], 'cnt': cnt, 'area': area, 'mass': calcular_cbuq_kg(area), 'conf': float(r.boxes[i].conf[0])})
        elif r.boxes:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if (x2-x1)*(y2-y1) < 500: continue
                area = calcular_area_cm2((x2-x1)*(y2-y1))
                raw_buracos.append({'box': [x1,y1,x2,y2], 'cnt': None, 'area': area, 'mass': calcular_cbuq_kg(area), 'conf': float(box.conf[0])})

    # 3. CONFLITOS
    remover_bueiro_idx = set()
    remover_buraco_idx = set()

    for i, bueiro in enumerate(raw_bueiros):
        for j, buraco in enumerate(raw_buracos):
            if get_iou(bueiro['box'], buraco['box']) > 0.1:
                if bueiro['label'] in ['Quebrado', 'Aberto', 'Sem Tampa']:
                    remover_buraco_idx.add(j)
                elif bueiro['label'] == 'Bom':
                    remover_bueiro_idx.add(i)

    # 4. DESENHO FINAL
    count_obj = 0
    total_dano_cm2 = 0
    total_cbuq_kg = 0

    # Desenha Buracos
    for i, obj in enumerate(raw_buracos):
        if i in remover_buraco_idx: continue
        
        count_obj += 1
        total_dano_cm2 += obj['area']
        total_cbuq_kg += obj['mass']
        
        if obj['cnt'] is not None: cv2.fillPoly(overlay, [obj['cnt']], COR_BURACO)
        
        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(img_original, (x1,y1), (x2,y2), COR_BURACO, THICKNESS)
        desenhar_etiqueta(img_original, f"Buraco {count_obj}", x1, y1, COR_BURACO, (0,0,0), 'topo')
        
        relatorio.append({
            "id": f"#{count_obj}", "tipo": "Buraco", "detalhe": f"{(obj['area']/10000):.4f} m²",
            "extra": f"{(obj['mass']/1000):.3f} t", "conf": f"{obj['conf']:.2f}"
        })

    # Desenha Bueiros
    bueiros_finais = []
    for i, b in enumerate(raw_bueiros):
        if i in remover_bueiro_idx: continue
        
        duplicado = False
        for existente in bueiros_finais:
            if get_iou(b['box'], existente['box']) > 0.3:
                duplicado = True
                if b['label'] == 'Quebrado' and existente['label'] in ['Aberto', 'Sem Tampa']: existente.update(b)
                break
        if not duplicado: bueiros_finais.append(b)

    for obj in bueiros_finais:
        count_obj += 1
        x1,y1,x2,y2 = obj['box']
        color = CORES_BUEIRO.get(obj['label'], (255,0,0))
        txt_color = (0,0,0) if obj['label'] == 'Bom' else (255,255,255)
        
        cv2.rectangle(img_original, (x1,y1), (x2,y2), color, THICKNESS)
        desenhar_etiqueta(img_original, obj['label'], x1, y2, color, txt_color, 'fundo')

        relatorio.append({
            "id": f"#{count_obj}", "tipo": "Bueiro", "detalhe": obj['label'],
            "extra": "-", "conf": f"{obj['conf']:.2f}"
        })

    if detection_mode == 'mask': img_final = cv2.addWeighted(overlay, 0.3, img_original, 0.7, 0)
    else: img_final = img_original

    _, b64 = cv2.imencode('.jpg', img_final)
    
    return {
        "file_name": file_name, "image_b64": base64.b64encode(b64).decode('utf-8'),
        "report_data": relatorio, "count": count_obj,
        "total_dano_m2": total_dano_cm2/10000.0, "total_cbuq_t": total_cbuq_kg/1000.0
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar")
async def detectar(
    files: list[UploadFile] = File(...),
    confidence_buraco: float = Form(0.25),
    confidence_bueiro: float = Form(0.25),
    detection_mode: str = Form("box")
):
    results, grand_m2, grand_ton, grand_cnt = [], 0, 0, 0
    for f in files:
        data = await f.read()
        res = processar_arquivo_unico(data, f.filename, confidence_buraco, confidence_bueiro, detection_mode)
        if res:
            results.append(res)
            grand_m2 += res['total_dano_m2']
            grand_ton += res['total_cbuq_t']
            grand_cnt += res['count']
            
    return JSONResponse({
        "batch_results": results,
        "grand_totals": {"m2": f"{grand_m2:.4f}", "ton": f"{grand_ton:.3f}", "count": grand_cnt}
    })
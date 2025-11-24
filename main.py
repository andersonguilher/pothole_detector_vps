from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import base64
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates_static")

# --- CARREGAMENTO DOS MODELOS ---
models = {}
print("-" * 30)
try: models["box"] = YOLO("models/buracos_box.pt"); print("✅ buracos_box.pt (Buracos) carregado.")
except: print("❌ Erro ao carregar buracos_box.pt")

try: models["mask"] = YOLO("models/buracos_seg.pt"); print("✅ buracos_seg.pt (Mask) carregado.")
except: print("⚠️ buracos_seg.pt não encontrado.")

try: models["bueiro"] = YOLO("models/bueiros_det.pt"); print("✅ bueiros_det.pt (Bueiros) carregado.")
except: print("❌ bueiros_det.pt não encontrado.")
print("-" * 30)

# --- CONFIGURAÇÕES GERAIS ---
RESOLUCAO_IA = 736

# TRADUÇÃO
TRADUCAO_BUEIRO = {
    'good': 'TAMPÃO (BOM)', 
    'broke': 'TAMPÃO (QUEBRADO)', 
    'uncover': 'BUEIRO (ABERTO)',
    'missing': 'BUEIRO (SEM TAMPA)'
}

# CORES
CORES_BUEIRO = {
    'TAMPÃO (BOM)': (0, 255, 0),      # Verde
    'TAMPÃO (QUEBRADO)': (0, 0, 255), # Vermelho
    'BUEIRO (ABERTO)': (0, 0, 255),   # Vermelho
    'BUEIRO (SEM TAMPA)': (0, 0, 255) # Vermelho
}
COR_BURACO = (0, 255, 255)   # Amarelo/Ciano

templates = Jinja2Templates(directory="templates")

# Configuração Visual
THICKNESS = 2
PADDING = 4

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

# NOVA FUNÇÃO DE FUSÃO ROBUSTA
def fundir_bueiros_recursivo(lista_bueiros):
    # Loop infinito que só para quando não houver mais nada para fundir
    while True:
        houve_fusao = False
        nova_lista = []
        indices_processados = set()

        for i in range(len(lista_bueiros)):
            if i in indices_processados: continue
            
            base = lista_bueiros[i]
            fundido = False

            # Compara com todos os outros itens seguintes
            for j in range(i + 1, len(lista_bueiros)):
                if j in indices_processados: continue
                
                candidato = lista_bueiros[j]
                
                # Se for o mesmo tipo e tiver QUALQUER sobreposição (> 0.01)
                if base['label'] == candidato['label'] and get_iou(base['box'], candidato['box']) > 0.01:
                    # Cria um super-box contendo os dois
                    novo_x1 = min(base['box'][0], candidato['box'][0])
                    novo_y1 = min(base['box'][1], candidato['box'][1])
                    novo_x2 = max(base['box'][2], candidato['box'][2])
                    novo_y2 = max(base['box'][3], candidato['box'][3])
                    
                    base['box'] = [novo_x1, novo_y1, novo_x2, novo_y2]
                    base['conf'] = max(base['conf'], candidato['conf'])
                    
                    indices_processados.add(j) # Marca o segundo como "já usado"
                    houve_fusao = True # Avisa que houve mudança, precisaremos rodar de novo
            
            nova_lista.append(base)
        
        lista_bueiros = nova_lista
        if not houve_fusao:
            break # Se rodou a lista toda e não fundiu nada, terminamos
            
    return lista_bueiros

def desenhar_etiqueta(img, texto, x, y, cor_fundo, cor_texto=(0,0,0), posicao='topo'):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_size = 14 
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), texto, font=font)
    w_txt = bbox[2] - bbox[0]
    h_txt = bbox[3] - bbox[1]
    pad = PADDING 

    h_img, _, _ = img.shape
    if posicao == 'topo':
        y_bg_top = max(0, y - h_txt - (pad * 2))
        y_bg_bottom = y
        pt_txt = (x + pad, y_bg_top + pad - 2)
    else:
        y_bg_top = y
        y_bg_bottom = min(h_img, y + h_txt + (pad * 2))
        pt_txt = (x + pad, y + pad - 2)

    draw.rectangle([x, y_bg_top, x + w_txt + (pad * 2), y_bg_bottom], fill=cor_fundo[::-1])
    draw.text(pt_txt, texto, font=font, fill=cor_texto[::-1])
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def processar_arquivo_unico(file_data, file_name, conf_buraco, conf_bueiro, detection_mode):
    nparr = np.frombuffer(file_data, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None: return None

    overlay = img_original.copy()
    relatorio = []
    
    raw_bueiros = [] 
    raw_buracos = []

    # 1. DETECÇÃO DE BUEIROS
    if "bueiro" in models:
        res_b = models["bueiro"].predict(
            source=img_original, save=False, conf=conf_bueiro, iou=0.5, verbose=False, imgsz=RESOLUCAO_IA
        )
        for box in res_b[0].boxes:
            cls_name = res_b[0].names[int(box.cls[0])]
            label_pt = TRADUCAO_BUEIRO.get(cls_name, cls_name)
            if label_pt == 'TAMPÃO (BOM)' and float(box.conf[0]) < 0.50: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            raw_bueiros.append({'box': [x1,y1,x2,y2], 'label': label_pt, 'conf': float(box.conf[0])})

    # --- NOVO: APLICA A FUSÃO DE BUEIROS IMEDIATAMENTE ---
    # Isso junta a tampa solta com o buraco ANTES de calcular conflitos
    raw_bueiros = fundir_bueiros_recursivo(raw_bueiros)

    # 2. DETECÇÃO DE BURACOS
    model_buraco = models.get("mask" if detection_mode == 'mask' else "box")
    if model_buraco:
        res_p = model_buraco.predict(
            source=img_original, save=False, conf=conf_buraco, iou=0.4, verbose=False, imgsz=RESOLUCAO_IA
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

    # 3. CONFLITOS (BURACO vs BUEIRO JÁ FUNDIDO)
    remover_bueiro_idx = set()
    remover_buraco_idx = set()

    for i, bueiro in enumerate(raw_bueiros):
        for j, buraco in enumerate(raw_buracos):
            if get_iou(bueiro['box'], buraco['box']) > 0.1:
                # Prioridade: Bueiro ruim anula buraco
                if bueiro['label'] in ['TAMPÃO (QUEBRADO)', 'BUEIRO (ABERTO)', 'BUEIRO (SEM TAMPA)']:
                    remover_buraco_idx.add(j)
                # Prioridade: Bueiro bom anula a si mesmo se tiver buraco (raro, mas mantém lógica)
                elif bueiro['label'] == 'TAMPÃO (BOM)':
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
        desenhar_etiqueta(img_original, f"BURACO {count_obj}", x1, y1, COR_BURACO, (0,0,0), 'topo')
        
        relatorio.append({
            "id": f"#{count_obj}", "tipo": "BURACO", "detalhe": f"{(obj['area']/10000):.4f} m²",
            "extra": f"{(obj['mass']/1000):.3f} t", "conf": f"{obj['conf']:.2f}"
        })

    # Desenha Bueiros (Agora a lista raw_bueiros já está limpa e fundida)
    bueiros_finais = []
    
    # Pequena verificação final para duplicados de tipos DIFERENTES (Ex: Aberto sobrepondo Quebrado)
    # A função recursiva já resolveu os de tipos IGUAIS.
    for i, b in enumerate(raw_bueiros):
        if i in remover_bueiro_idx: continue
        
        duplicado = False
        for existente in bueiros_finais:
            if get_iou(b['box'], existente['box']) > 0.3:
                duplicado = True
                # Mantém o mais grave
                if b['label'] == 'TAMPÃO (QUEBRADO)' and existente['label'] in ['BUEIRO (ABERTO)', 'BUEIRO (SEM TAMPA)']:
                    existente.update(b)
                break
        if not duplicado: bueiros_finais.append(b)

    for obj in bueiros_finais:
        count_obj += 1
        x1,y1,x2,y2 = obj['box']
        color = CORES_BUEIRO.get(obj['label'], (255,0,0))
        txt_color = (0,0,0) if obj['label'] == 'TAMPÃO (BOM)' else (255,255,255)
        
        cv2.rectangle(img_original, (x1,y1), (x2,y2), color, THICKNESS)
        desenhar_etiqueta(img_original, obj['label'], x1, y2, color, txt_color, 'fundo')

        relatorio.append({
            "id": f"#{count_obj}", "tipo": "BUEIRO", "detalhe": obj['label'],
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
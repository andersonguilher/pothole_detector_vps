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
import math

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

# Física
DENSIDADE_CBUQ_TM3 = 2.4 
ESPESSURA_CM = 5 

# --- FUNÇÕES ---

# FUNÇÕES DE CÁLCULO AGORA RECEBEM PIXEL_TO_CM2
def calcular_area_cm2(area_pixel, pixel_to_cm2): return area_pixel * pixel_to_cm2
def calcular_cbuq_kg(area_cm2): return (area_cm2 / 10000.0 * (ESPESSURA_CM / 100.0)) * DENSIDADE_CBUQ_TM3 * 1000

# Função para converter pixel para medida linear (cm ou m)
def pixel_to_linear_measure(pixels, pixel_to_cm2):
    PIXEL_TO_CM = math.sqrt(pixel_to_cm2)
    return pixels * PIXEL_TO_CM # Retorna em CM

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

# Função de fusão robusta (mantida)
def fundir_bueiros_recursivo(lista_bueiros):
    while True:
        houve_fusao = False
        nova_lista = []
        indices_processados = set()

        for i in range(len(lista_bueiros)):
            if i in indices_processados: continue
            
            base = lista_bueiros[i]
            
            for j in range(i + 1, len(lista_bueiros)):
                if j in indices_processados: continue
                
                candidato = lista_bueiros[j]
                
                if base['label'] == candidato['label'] and get_iou(base['box'], candidato['box']) > 0.01:
                    novo_x1 = min(base['box'][0], candidato['box'][0])
                    novo_y1 = min(base['box'][1], candidato['box'][1])
                    novo_x2 = max(base['box'][2], candidato['box'][2])
                    novo_y2 = max(base['box'][3], candidato['box'][3])
                    
                    base['box'] = [novo_x1, novo_y1, novo_x2, novo_y2]
                    base['conf'] = max(base['conf'], candidato['conf'])
                    
                    indices_processados.add(j) 
                    houve_fusao = True 
            
            nova_lista.append(base)
        
        lista_bueiros = nova_lista
        if not houve_fusao:
            break 
            
    return lista_bueiros

# FUNÇÃO DE DESENHO ESCALÁVEL (suporta '\n')
def desenhar_etiqueta(img, texto, x, y, cor_fundo, cor_texto=(0,0,0), posicao='topo', thickness=2):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    h_img, _, _ = img.shape
    
    text_lines = texto.split('\n')
    
    font_base = 20
    font_size = int(h_img / 720 * font_base)
    font_size = max(font_size, 12)
    
    pad = max(2, int(font_size / 6)) 

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
    # Calcula as dimensões do background
    max_w_txt = 0
    line_height = 0
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_w_txt = max(max_w_txt, bbox[2] - bbox[0])
        line_height = max(line_height, bbox[3] - bbox[1])
        
    w_txt = max_w_txt
    h_txt_total = line_height * len(text_lines) + (len(text_lines) - 1) * 2 # 2px de espaçamento entre linhas

    if posicao == 'topo':
        y_bg_top = max(0, y - h_txt_total - (pad * 2))
        y_bg_bottom = y
        pt_txt_y = y_bg_top + pad - 2
    else: # posicao == 'fundo'
        y_bg_top = y
        y_bg_bottom = min(h_img, y + h_txt_total + (pad * 2))
        pt_txt_y = y + pad - 2

    # Desenha o background
    draw.rectangle([x, y_bg_top, x + w_txt + (pad * 2), y_bg_bottom], fill=cor_fundo[::-1])
    
    # Desenha as linhas de texto separadamente
    current_y = pt_txt_y
    
    for line in text_lines:
        draw.text((x + pad, current_y), line, font=font, fill=cor_texto[::-1])
        current_y += line_height + 2 # Pula para a próxima linha

    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# FUNÇÃO PRINCIPAL AGORA RECEBE pixel_to_cm2 e measure_display_mode
def processar_arquivo_unico(file_data, file_name, conf_buraco, conf_bueiro, detection_mode, resolution_ia, calculate_measures, measure_display_mode, pixel_to_cm2):
    nparr = np.frombuffer(file_data, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None: return None

    h_orig, w_orig = img_original.shape[:2]

    # --- Aplica a divisão por 10 ao fator de área ---
    pixel_to_cm2 = pixel_to_cm2 / 10.0
    
    THICKNESS = max(2, int(h_orig / 720 * 3))
    
    overlay = img_original.copy()
    relatorio = []
    
    raw_bueiros = [] 
    raw_buracos = []

    # 1. DETECÇÃO DE BUEIROS 
    if "bueiro" in models:
        res_b = models["bueiro"].predict(
            source=img_original, save=False, conf=conf_bueiro, iou=0.5, verbose=False, imgsz=resolution_ia
        )
        for box in res_b[0].boxes:
            cls_name = res_b[0].names[int(box.cls[0])]
            label_pt = TRADUCAO_BUEIRO.get(cls_name, cls_name)
            if label_pt == 'TAMPÃO (BOM)' and float(box.conf[0]) < 0.50: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            raw_bueiros.append({'box': [x1,y1,x2,y2], 'label': label_pt, 'conf': float(box.conf[0])})

    raw_bueiros = fundir_bueiros_recursivo(raw_bueiros)

    # 2. DETECÇÃO DE BURACOS 
    model_buraco = models.get("mask" if detection_mode == 'mask' else "box")
    if model_buraco:
        res_p = model_buraco.predict(
            source=img_original, save=False, conf=conf_buraco, iou=0.4, verbose=False, imgsz=resolution_ia
        )
        r = res_p[0]
        if detection_mode == 'mask' and r.masks:
            for i, mask in enumerate(r.masks):
                cnt = mask.xy[0].astype(np.int32)
                if cv2.contourArea(cnt) < 500: continue
                x,y,w,h = cv2.boundingRect(cnt)
                area = calcular_area_cm2(cv2.contourArea(cnt), pixel_to_cm2) if calculate_measures else 0
                mass = calcular_cbuq_kg(area) if calculate_measures else 0
                raw_buracos.append({'box': [x,y,x+w,y+h], 'cnt': cnt, 'area': area, 'mass': mass, 'conf': float(r.boxes[i].conf[0])})
        elif r.boxes:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                box_area_pixel = (x2-x1)*(y2-y1)
                if box_area_pixel < 500: continue
                area = calcular_area_cm2(box_area_pixel, pixel_to_cm2) if calculate_measures else 0
                mass = calcular_cbuq_kg(area) if calculate_measures else 0
                raw_buracos.append({'box': [x1,y1,x2,y2], 'cnt': None, 'area': area, 'mass': mass, 'conf': float(box.conf[0])})

    # 3. CONFLITOS 
    remover_bueiro_idx = set()
    remover_buraco_idx = set()

    for i, bueiro in enumerate(raw_bueiros):
        for j, buraco in enumerate(raw_buracos):
            if get_iou(bueiro['box'], buraco['box']) > 0.1:
                if bueiro['label'] in ['TAMPÃO (QUEBRADO)', 'BUEIRO (ABERTO)', 'BUEIRO (SEM TAMPA)']:
                    remover_buraco_idx.add(j)
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
        
        prefixo = f"#{count_obj} BURACO"
        
        if calculate_measures and measure_display_mode != 'none':
            # --- CÁLCULO DE DIMENSÕES ---
            w_pixels = x2 - x1
            h_pixels = y2 - y1
            
            w_m = pixel_to_linear_measure(w_pixels, pixel_to_cm2) / 100.0
            h_m = pixel_to_linear_measure(h_pixels, pixel_to_cm2) / 100.0
            area_m2 = obj['area'] / 10000.0
            
            dim_str = f"{w_m:.2f} m x {h_m:.2f} m".replace('.', ',')
            area_str = f"{area_m2:.2f} m²".replace('.', ',')
            
            # --- CONSTRUÇÃO DA SEGUNDA LINHA BASEADA NO MODO DE EXIBIÇÃO ---
            if measure_display_mode == 'dimensions':
                label_text_measures = dim_str
            elif measure_display_mode == 'area':
                label_text_measures = area_str
            elif measure_display_mode == 'both':
                label_text_measures = f"{dim_str} = {area_str}"
            
            if measure_display_mode == 'none':
                 # O caso 'none' não deve ser processado aqui se calculate_measures=True
                 # Mas se for, ele cairá no else de label_text abaixo.
                 pass

            # Determina o texto final da etiqueta
            if measure_display_mode in ['dimensions', 'area', 'both']:
                # Formato completo com quebra de linha
                 label_text = f"{prefixo}\n{label_text_measures}"
            else:
                 # *** ALTERAÇÃO AQUI: Formato simplificado com % na mesma linha ***
                 conf_percent = f"{obj['conf'] * 100:.0f}%"
                 label_text = f"{prefixo} {conf_percent}"


            detalhe_buraco = f"{area_m2:.4f} m²"
            extra_buraco = f"{(obj['mass']/1000):.3f} t"
        else:
            # Se calculate_measures=False (ou measure_display_mode='none' e calculate_measures=True)
            # *** ALTERAÇÃO AQUI: Formato simplificado com % na mesma linha ***
            conf_percent = f"{obj['conf'] * 100:.0f}%"
            label_text = f"{prefixo} {conf_percent}"
            detalhe_buraco = f"{obj['conf']:.2f}"
            extra_buraco = "-"
        
        # Desenha a etiqueta 
        desenhar_etiqueta(img_original, label_text, x1, y1, COR_BURACO, (0,0,0), 'topo', THICKNESS) 
        
        relatorio.append({
            "id": f"#{count_obj}", "tipo": "BURACO", "detalhe": detalhe_buraco,
            "extra": extra_buraco, "conf": f"{obj['conf']:.2f}"
        })

    # Desenha Bueiros 
    bueiros_finais = []
    
    for i, b in enumerate(raw_bueiros):
        if i in remover_bueiro_idx: continue
        
        duplicado = False
        for existente in bueiros_finais:
            if get_iou(b['box'], existente['box']) > 0.3:
                duplicado = True
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
        
        # *** ALTERAÇÃO AQUI: Formato simplificado para bueiro ***
        bueiro_label_status = obj['label']
        conf_percent = f"{obj['conf'] * 100:.0f}%"
        
        # Etiqueta do bueiro sempre mostra o label (tipo) + confiança
        bueiro_label_final = f"#{count_obj} {bueiro_label_status}\n{conf_percent}"
        
        # Verifica se o modo de exibição de medidas está ativo, mas o modo de exibição é Confiança Apenas
        # Se measure_display_mode != 'none', ele exibe o status + a confiança abaixo.
        # Se measure_display_mode == 'none' (ou calculate_measures=False), ele deve exibir o status e a confiança na mesma linha.

        if calculate_measures and measure_display_mode in ['dimensions', 'area', 'both']:
            # Exibir status (tipo) na linha 1 e Confiança na linha 2 (como antes)
            bueiro_label_final = f"#{count_obj} {bueiro_label_status}\nConfiança: {obj['conf']:.2f}" # Usando conf .2f para detalhe
        else:
            # *** ALTERAÇÃO AQUI: Formato simplificado para bueiro na mesma linha ***
            bueiro_label_final = f"#{count_obj} {bueiro_label_status} {conf_percent}"
        
        
        desenhar_etiqueta(img_original, bueiro_label_final, x1, y2, color, txt_color, 'fundo', THICKNESS)

        extra_bueiro = "-"
        if not calculate_measures:
             extra_bueiro = f"{obj['conf']:.2f}"
             
        relatorio.append({
            "id": f"#{count_obj}", "tipo": "BUEIRO", "detalhe": obj['label'],
            "extra": extra_bueiro, "conf": f"{obj['conf']:.2f}"
        })

    if detection_mode == 'mask': img_final = cv2.addWeighted(overlay, 0.3, img_original, 0.7, 0)
    else: img_final = img_original

    _, b64 = cv2.imencode('.jpg', img_final)
    
    total_m2_res = total_dano_cm2/10000.0 if calculate_measures else 0.0
    total_t_res = total_cbuq_kg/1000.0 if calculate_measures else 0.0
    
    return {
        "file_name": file_name, 
        "image_b64": base64.b64encode(b64).decode('utf-8'),
        "report_data": relatorio, 
        "count": count_obj,
        "total_dano_m2": total_m2_res, 
        "total_cbuq_t": total_t_res
    }
    
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar")
async def detectar(
    files: list[UploadFile] = File(...),
    confidence_buraco: float = Form(0.25),
    confidence_bueiro: float = Form(0.30),
    detection_mode: str = Form("box"),
    resolution_ia: int = Form(736),
    pixel_to_cm2: float = Form(0.04), 
    calculate_measures: str = Form("true"),
    measure_display_mode: str = Form("both")
):
    is_calculating_measures = calculate_measures.lower() == 'true'
    
    results, grand_m2, grand_ton, grand_cnt = [], 0, 0, 0
    for f in files:
        data = await f.read()
        res = processar_arquivo_unico(
            data, f.filename, confidence_buraco, confidence_bueiro, detection_mode, resolution_ia, is_calculating_measures, measure_display_mode, pixel_to_cm2
        )
        if res:
            results.append(res)
            grand_m2 += res['total_dano_m2']
            grand_ton += res['total_cbuq_t']
            grand_cnt += res['count']
            
    return JSONResponse({
        "batch_results": results,
        "grand_totals": {"m2": f"{grand_m2:.4f}", "ton": f"{grand_ton:.3f}", "count": grand_cnt}
    })
from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import base64
from PIL import Image, ImageDraw, ImageFont
import math
import tempfile
import os
import time
import json
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# TRADUÇÃO E CORES
TRADUCAO_BUEIRO = { 'good': 'TAMPÃO BOM', 'broke': 'TAMPÃO QUEBRADO', 'uncover': 'BUEIRO ABERTO', 'missing': 'SEM TAMPA' }
CORES_MAPA = {
    'BURACO': (0, 255, 255),          # Amarelo
    'TAMPÃO BOM': (0, 255, 0),        # Verde
    'TAMPÃO QUEBRADO': (0, 0, 255),   # Vermelho
    'BUEIRO ABERTO': (0, 0, 255),     # Vermelho
    'SEM TAMPA': (0, 0, 255)          # Vermelho
}

templates = Jinja2Templates(directory="templates")

# Física
DENSIDADE_CBUQ_TM3 = 2.4 
ESPESSURA_CM = 5 

# --- VARIÁVEL GLOBAL DE SUPRESSÃO ---
CAPTURE_HISTORY = []

# --- FUNÇÕES AUXILIARES ---
def calcular_area_cm2(area_pixel, pixel_to_cm2): return area_pixel * pixel_to_cm2
def calcular_cbuq_kg(area_cm2): return (area_cm2 / 10000.0 * (ESPESSURA_CM / 100.0)) * DENSIDADE_CBUQ_TM3 * 1000
def pixel_to_linear_measure(pixels, pixel_to_cm2): return pixels * math.sqrt(pixel_to_cm2)

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def get_center_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2; cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2; cyB = (boxB[1] + boxB[3]) / 2
    return math.hypot(cxA - cxB, cyA - cyB)

def is_duplicate_capture(cx, cy, current_time, threshold_px=150, cooldown_sec=3.0):
    global CAPTURE_HISTORY
    capture_history = [c for c in CAPTURE_HISTORY if current_time - c['time'] < cooldown_sec]
    for c in capture_history:
        dist = math.hypot(cx - c['x'], cy - c['y'])
        if dist < threshold_px: return True
    return False

def fundir_bueiros_recursivo(lista_bueiros):
    while True:
        houve_fusao = False; nova_lista = []; indices_processados = set()
        for i in range(len(lista_bueiros)):
            if i in indices_processados: continue
            base = lista_bueiros[i]
            for j in range(i + 1, len(lista_bueiros)):
                if j in indices_processados: continue
                candidato = lista_bueiros[j]
                if base['label'] == candidato['label'] and get_iou(base['box'], candidato['box']) > 0.01:
                    base['box'] = [min(base['box'][0], candidato['box'][0]), min(base['box'][1], candidato['box'][1]), max(base['box'][2], candidato['box'][2]), max(base['box'][3], candidato['box'][3])]
                    base['conf'] = max(base['conf'], candidato['conf'])
                    indices_processados.add(j); houve_fusao = True
            nova_lista.append(base)
        lista_bueiros = nova_lista
        if not houve_fusao: break 
    return lista_bueiros

def desenhar_etiqueta(img, texto, x, y, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("arial.ttf", 14) 
    except: font = ImageFont.load_default()
    lines = texto.split('\n')
    max_w = 0; total_h = 0; line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1] + 4
        max_w = max(max_w, w); total_h += h; line_heights.append(h)
    pad = 4; text_x = x; text_y = y - total_h - (pad * 2)
    if text_y < 0: text_y = y 
    bg_color_rgb = (color[2], color[1], color[0])
    brightness = (bg_color_rgb[0] * 299 + bg_color_rgb[1] * 587 + bg_color_rgb[2] * 114) / 1000
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw.rectangle([text_x, text_y, text_x + max_w + (pad * 2), text_y + total_h + (pad * 2)], fill=bg_color_rgb)
    curr_y = text_y + pad
    for i, line in enumerate(lines):
        draw.text((text_x + pad, curr_y), line, font=font, fill=text_color)
        curr_y += line_heights[i]
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def filtrar_por_roi(deteccoes, width, height, polygon_points=None):
    if not deteccoes: return [], polygon_points
    if polygon_points is None:
        x_bottom_left = 0; x_top_left = int(width * 0.35); x_top_right = int(width * 0.65); x_bottom_right = width
        y_bottom = height; y_top = int(height * 0.35)
        p1 = (x_bottom_left, y_bottom); p2 = (x_top_left, y_top); p3 = (x_top_right, y_top); p4 = (x_bottom_right, y_bottom)
        polygon_points = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    deteccoes_validas = []
    for det in deteccoes:
        x1, y1, x2, y2 = det['box']; cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        if cv2.pointPolygonTest(polygon_points, (cx, cy), False) >= 0: deteccoes_validas.append(det)
    return deteccoes_validas, polygon_points

# --- CORE: PROCESSAMENTO DE FRAME ---
def processar_frame(
    img_original, conf_buraco, conf_bueiro, detection_mode, resolution_ia, calculate_measures, pixel_to_cm2, 
    active_tracks, next_object_id, visual_buffer, 
    use_roi=False, show_trigger_line=False, 
    show_id=True, show_type=True, show_conf=True, show_dim=True, show_area=True, 
    capture_seq_ref=None,
    filter_by_size=False, target_size=0.90
):
    global CAPTURE_HISTORY
    h_orig, w_orig = img_original.shape[:2]; overlay = img_original.copy()
    relatorio = []; raw_detections = []; generated_thumbnails = [] 
    area_val = 0; w_m = 0; h_m = 0; x1=0; y1=0; x2=0; y2=0; cx=0; cy=0
    
    # 1. Inferência
    if "bueiro" in models:
        res_b = models["bueiro"].predict(source=img_original, save=False, conf=conf_bueiro, iou=0.5, verbose=False, imgsz=resolution_ia)
        for box in res_b[0].boxes:
            label_pt = TRADUCAO_BUEIRO.get(res_b[0].names[int(box.cls[0])], res_b[0].names[int(box.cls[0])])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            raw_detections.append({'box': [x1,y1,x2,y2], 'label': label_pt, 'conf': float(box.conf[0]), 'type': 'BUEIRO'})
    
    if models.get(detection_mode == 'mask' and "mask" or "box"):
        model = models["mask"] if detection_mode == 'mask' else models["box"]
        res_p = model.predict(source=img_original, save=False, conf=conf_buraco, iou=0.4, verbose=False, imgsz=resolution_ia)
        r = res_p[0]
        if detection_mode == 'mask' and r.masks:
            for i, mask in enumerate(r.masks):
                cnt = mask.xy[0].astype(np.int32)
                if cv2.contourArea(cnt) < 300: continue
                x,y,w,h = cv2.boundingRect(cnt)
                area = calcular_area_cm2(cv2.contourArea(cnt), pixel_to_cm2) if calculate_measures else 0
                mass = calcular_cbuq_kg(area) if calculate_measures else 0
                raw_detections.append({'box': [x,y,x+w,y+h], 'label': 'BURACO', 'conf': float(r.boxes[i].conf[0]), 'type': 'BURACO', 'cnt': cnt, 'area': area, 'mass': mass})
        elif r.boxes:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                box_area_pix = (x2-x1)*(y2-y1)
                if box_area_pix < 300: continue
                area = calcular_area_cm2(box_area_pix, pixel_to_cm2) if calculate_measures else 0
                mass = calcular_cbuq_kg(area) if calculate_measures else 0
                raw_detections.append({'box': [x1,y1,x2,y2], 'label': 'BURACO', 'conf': float(box.conf[0]), 'type': 'BURACO', 'cnt': None, 'area': area, 'mass': mass})

    # 2. FILTRAGEM (ROI + TAMANHO + PARALAXE)
    deteccoes_validas = []
    
    if use_roi:
        candidatos = filtrar_por_roi(raw_detections, w_orig, h_orig)[0]
    else:
        candidatos = raw_detections

    for det in candidatos:
        # Cálculos de Dimensão ANTES do tracking para poder filtrar
        bx = det['box']
        wb = bx[2] - bx[0]
        hb = bx[3] - bx[1]
        
        # --- CORREÇÃO DE PARALAXE ---
        # A câmera "achata" objetos. Quanto maior a largura comparada à altura, mais achatado.
        # Se for um BUEIRO (que sabemos ser +/- quadrado ou redondo), assumimos que a Largura é a medida real.
        # Para BURACOS, tentamos corrigir a altura baseada num fator empírico de inclinação (~2.5x).
        
        wm_m = pixel_to_linear_measure(wb, pixel_to_cm2)/100.0
        hm_m = pixel_to_linear_measure(hb, pixel_to_cm2)/100.0
        
        if det['type'] == 'BUEIRO':
            # Assumimos bueiro quadrado/redondo: A medida real é a maior das duas (geralmente a largura na imagem)
            real_dim = max(wm_m, hm_m)
            # Forçamos as duas medidas a serem iguais para exibição correta (ex: 0.92 x 0.92)
            det['w_real'] = real_dim
            det['h_real'] = real_dim
        else:
            # Para buracos, aplicamos um fator de correção na altura (compensação de achatamento)
            # Fator 2.0 é uma aproximação comum para câmeras de painel nessa angulação
            det['w_real'] = wm_m
            det['h_real'] = hm_m * 2.0 
            
            # Recalcula área aproximada com a nova altura
            det['area_real'] = det['w_real'] * det['h_real']

        # Filtro de Tamanho (SÓ PARA BUEIROS)
        aceitar = True
        if det['type'] == 'BUEIRO' and filter_by_size:
            tolerance = 0.04 
            min_d = target_size * (1.0 - tolerance)
            max_d = target_size * (1.0 + tolerance)
            
            # Verifica se a dimensão real corrigida está no alvo
            if not (min_d <= det['w_real'] <= max_d):
                aceitar = False
        
        if aceitar:
            deteccoes_validas.append(det)

    # 3. Tracking
    new_active_tracks = []
    for det in deteccoes_validas:
        best_idx = -1; best_dist = float('inf')
        for i, track in enumerate(active_tracks):
            dist = get_center_distance(det['box'], track['box'])
            if det['label'] == track['label'] and dist < (w_orig * 0.15):
                if dist < best_dist: best_dist = dist; best_idx = i
        
        if best_idx != -1:
            track = active_tracks.pop(best_idx)
            # Atualiza dados e DIMENSÕES CORRIGIDAS
            track.update({'box': det['box'], 'conf': det['conf'], 'data': det, 'frames_missing': 0})
            # Persiste as dimensões reais calculadas no filtro
            track['w_real'] = det['w_real']
            track['h_real'] = det['h_real']
            if 'area_real' in det: track['area_real'] = det['area_real']
            new_active_tracks.append(track)
        else:
            new_track = {'id': next_object_id, 'box': det['box'], 'label': det['label'], 'type': det['type'], 'conf': float(det['conf']), 'data': det, 'frames_missing': 0, 'thumb_sent': False}
            new_track['w_real'] = det['w_real']
            new_track['h_real'] = det['h_real']
            if 'area_real' in det: new_track['area_real'] = det['area_real']
            new_active_tracks.append(new_track)
            next_object_id += 1
            
    for track in active_tracks:
        track['frames_missing'] += 1
        if track['frames_missing'] < 10: new_active_tracks.append(track)
    active_tracks = new_active_tracks

    # 4. Captura e Desenho
    TRIGGER_Y = h_orig * 0.60 
    curr_time = time.time()

    for track in active_tracks:
        if track['frames_missing'] > 0: continue
        box = track['box']; x1,y1,x2,y2 = box; obj_data = track['data']
        center_x = (x1 + x2) // 2; center_y = (y1 + y2) // 2
        cx = center_x; cy = center_y
        
        color = CORES_MAPA.get(track['label'], (255,0,255))
        if track['type'] == 'BURACO': color = CORES_MAPA['BURACO']
        
        # --- MONTAGEM DA ETIQUETA (USANDO DIMENSÕES CORRIGIDAS) ---
        lines = []; header = []
        display_id = track.get('capture_id', track['id'])
        
        if track.get('thumb_sent', False) and show_id: header.append(f"#{track['capture_id']}")
        elif not track.get('thumb_sent', False) and show_id: header.append(f"#{track['id']}") 
        
        if show_type: header.append(f"{track['label']}")
        if show_conf: header.append(f"{track['conf']:.2f}")
        if header: lines.append(" ".join(header))
        
        if calculate_measures:
            # Usa as medidas já corrigidas armazenadas no track
            w_m = track.get('w_real', 0)
            h_m = track.get('h_real', 0)
            
            line2 = []
            if show_dim: line2.append(f"{w_m:.2f}x{h_m:.2f}m".replace('.',','))
            
            if track['type'] == 'BURACO' and show_area:
                 a_m2 = track.get('area_real', 0)
                 prefix = "= " if show_dim else ""
                 line2.append(f"{prefix}{a_m2:.2f}m²".replace('.',','))
            
            if line2: lines.append("".join(line2))

        final_label = "\n".join(lines)

        # A. Captura Automática
        is_visible = (y1 > 40 and y2 < h_orig - 10 and x1 > 5 and x2 < w_orig - 5)

        if capture_seq_ref is not None and cy > TRIGGER_Y and is_visible and not track.get('thumb_sent', False):
            if not is_duplicate_capture(cx, cy, curr_time, 150, 3.0):
                track['capture_id'] = capture_seq_ref[0]
                capture_seq_ref[0] += 1
                
                lines_cap = []
                if show_id: lines_cap.append(f"#{track['capture_id']}")
                if show_type: lines_cap.append(f"{track['label']}")
                if show_conf: lines_cap.append(f"{track['conf']:.2f}")
                lines_cap = [" ".join(lines_cap)]
                if len(lines) > 1: lines_cap.append(lines[1])
                final_label_cap = "\n".join(lines_cap)
                
                tw, th = 640, int(640/w_orig * h_orig)
                thumb = cv2.resize(img_original, (tw, th))
                tx1, ty1, tx2, ty2 = int(x1*(tw/w_orig)), int(y1*(th/h_orig)), int(x2*(tw/w_orig)), int(y2*(th/h_orig))
                cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), color, 2)
                desenhar_etiqueta(thumb, final_label_cap, tx1, ty1, color)
                
                _, buf = cv2.imencode('.jpg', thumb)
                b64 = base64.b64encode(buf).decode('utf-8')
                generated_thumbnails.append({"id": track['capture_id'], "label": track['label'], "image": b64})
                
                track['thumb_sent'] = True; CAPTURE_HISTORY.append({'x': cx, 'y': cy, 'time': curr_time})

                ov_w = int(w_orig * 0.20); ov_h = int(ov_w / w_orig * h_orig)
                thumb_ov = cv2.resize(img_original, (ov_w, ov_h))
                ox1, oy1, ox2, oy2 = int(x1*(ov_w/w_orig)), int(y1*(ov_h/h_orig)), int(x2*(ov_w/w_orig)), int(y2*(ov_h/h_orig))
                cv2.rectangle(thumb_ov, (ox1, oy1), (ox2, oy2), color, 2)
                desenhar_etiqueta(thumb_ov, f"#{track['capture_id']}", ox1, oy1, color)
                cv2.rectangle(thumb_ov, (0,0), (ov_w-1, ov_h-1), (255,255,255), 1)
                visual_buffer.insert(0, thumb_ov)
                if len(visual_buffer) > 4: visual_buffer.pop()

        # B. Desenho no Vídeo
        if track['type'] == 'BURACO' and track['data'].get('cnt') is not None:
            cv2.fillPoly(overlay, [track['data']['cnt']], color)
        
        cv2.rectangle(img_original, (x1, y1), (x2, y2), color, 2)
        desenhar_etiqueta(img_original, final_label, x1, y1, color)

        if track.get('capture_id'):
             relatorio.append({"id": f"#{track['capture_id']}", "tipo": track['label'], "conf": f"{track['conf']:.2f}", "area": f"{area_val:.2f}"})

    if show_trigger_line:
        y_trig = int(TRIGGER_Y)
        cv2.line(img_original, (0, int(TRIGGER_Y)), (w_orig, int(TRIGGER_Y)), (0,0,255), 2)
        cv2.putText(img_original, "LINHA 60%", (10, int(TRIGGER_Y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    if detection_mode == 'mask':
        cv2.addWeighted(overlay, 0.3, img_original, 0.7, 0, img_original)

    margin = 10; curr_y_ov = margin
    for th_ov in visual_buffer:
        h_ov, w_ov = th_ov.shape[:2]
        if curr_y_ov + h_ov > h_orig: break
        x_pos = w_orig - w_ov - margin
        roi = img_original[curr_y_ov:curr_y_ov+h_ov, x_pos:x_pos+w_ov]
        cv2.addWeighted(th_ov, 0.9, roi, 0.1, 0, roi)
        cv2.rectangle(img_original, (x_pos, curr_y_ov), (x_pos+w_ov, curr_y_ov+h_ov), (200,200,200), 1)
        curr_y_ov += h_ov + margin

    return img_original, relatorio, active_tracks, next_object_id, generated_thumbnails

# --- ROTAS ---
def processar_arquivo_unico(file_data, file_name, conf_buraco, conf_bueiro, detection_mode, resolution_ia, calculate_measures, pixel_to_cm2, show_id, show_type, show_conf, show_dim, show_area, filter_by_size, target_size):
    nparr = np.frombuffer(file_data, np.uint8); img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None: return None
    capture_seq_ref = [1]
    img_final, _, _, _, thumbnails = processar_frame(
        img_original, conf_buraco, conf_bueiro, detection_mode, resolution_ia, 
        calculate_measures, pixel_to_cm2 / 10.0, [], 1, [], use_roi=False, show_trigger_line=False,
        show_id=show_id, show_type=show_type, show_conf=show_conf, show_dim=show_dim, show_area=show_area,
        capture_seq_ref=capture_seq_ref, filter_by_size=filter_by_size, target_size=target_size
    )
    total_capturas = capture_seq_ref[0] - 1
    _, b64 = cv2.imencode('.jpg', img_final)
    return { "file_name": file_name, "image_b64": base64.b64encode(b64).decode('utf-8'), "thumbnails": thumbnails, "count": total_capturas }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar")
async def detectar(files: list[UploadFile] = File(...), conf_buraco: float = Form(0.25), conf_bueiro: float = Form(0.30), detection_mode: str = Form("box"), resolution_ia: int = Form(736), pixel_to_cm2: float = Form(0.04), calculate_measures: str = Form("true"),
                   show_id: str = Form("true"), show_type: str = Form("true"), show_conf: str = Form("true"), show_dim: str = Form("true"), show_area: str = Form("true"),
                   filter_by_size: str = Form("false"), target_size: float = Form(0.90)):
    results = []; total_captured = 0
    global CAPTURE_HISTORY; CAPTURE_HISTORY = []
    for f in files:
        data = await f.read()
        res = processar_arquivo_unico(data, f.filename, conf_buraco, conf_bueiro, detection_mode, resolution_ia, (calculate_measures.lower() == 'true'), pixel_to_cm2,
                                      (show_id.lower()=='true'), (show_type.lower()=='true'), (show_conf.lower()=='true'), (show_dim.lower()=='true'), (show_area.lower()=='true'),
                                      (filter_by_size.lower()=='true'), target_size)
        if res: results.append(res); total_captured += res.get('count', 0)
        
    return JSONResponse({"batch_results": results, "grand_totals": {"count": total_captured}})

@app.post("/detectar_video")
async def detectar_video(
    request: Request, video: UploadFile = File(...), conf_buraco: float = Form(0.25), conf_bueiro: float = Form(0.30), detection_mode: str = Form("box"), resolution_ia: int = Form(736), pixel_to_cm2: float = Form(0.04), 
    calculate_measures: str = Form("true"), show_trigger_line: bool = Form(False), show_id: bool = Form(True), show_type: bool = Form(True), show_conf: bool = Form(True), show_dim: bool = Form(True), show_area: bool = Form(True),
    filter_by_size: str = Form("false"), target_size: float = Form(0.90)
):
    global CAPTURE_HISTORY; CAPTURE_HISTORY = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.split('.')[-1]}") as tmp_input:
            video_content = await video.read(); tmp_input.write(video_content); input_path = tmp_input.name
    except Exception as e: return JSONResponse({"message": f"Erro IO: {e}"}, 500)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): os.unlink(input_path); return JSONResponse({"message": "Erro ao abrir vídeo."}, 500)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT_WIDTH = 1280; aspect_ratio = height / width; OUTPUT_HEIGHT = int(OUTPUT_WIDTH * aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'avc1'); temp_output_filename = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(temp_output_filename, fourcc, fps_video, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened(): out.release(); fourcc = cv2.VideoWriter_fourcc(*'mp4v'); temp_output_filename = tempfile.mktemp(suffix=".mp4"); out = cv2.VideoWriter(temp_output_filename, fourcc, fps_video, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    async def process_stream():
        frame_current = 0; start_time = time.time(); active_tracks = []; next_object_id = 1; client_disconnected = False; visual_buffer = []
        capture_seq_ref = [1]
        captured_count = 0
        
        try:
            while cap.isOpened():
                if await request.is_disconnected(): client_disconnected = True; break
                ret, frame = cap.read()
                if not ret: break
                
                frame_det, _, active_tracks, next_object_id, new_thumbnails = processar_frame(
                    frame, conf_buraco, conf_bueiro, detection_mode, resolution_ia,
                    (calculate_measures.lower()=='true'), pixel_to_cm2/10.0, active_tracks, next_object_id, visual_buffer, 
                    use_roi=True, show_trigger_line=show_trigger_line,
                    show_id=show_id, show_type=show_type, show_conf=show_conf, show_dim=show_dim, show_area=show_area,
                    capture_seq_ref=capture_seq_ref,
                    filter_by_size=(filter_by_size.lower()=='true'), target_size=target_size
                )
                
                captured_count += len(new_thumbnails)
                out.write(cv2.resize(frame_det, (OUTPUT_WIDTH, OUTPUT_HEIGHT)))
                frame_current += 1
                
                if frame_current % 3 == 0 or frame_current == total_frames or len(new_thumbnails) > 0:
                    h_prev, w_prev = frame_det.shape[:2]; scale_prev = 640 / w_prev 
                    _, buffer = cv2.imencode('.jpg', cv2.resize(frame_det, (640, int(h_prev * scale_prev))), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    progress_data = { "status": "processing", "current": frame_current, "total": total_frames, "objs": captured_count, "preview": base64.b64encode(buffer.tobytes()).decode('utf-8'), "thumbnails": new_thumbnails }
                    try: yield json.dumps(progress_data) + "\n"
                    except: client_disconnected = True; break
            
            cap.release(); out.release(); cv2.destroyAllWindows()
            if os.path.exists(input_path): os.unlink(input_path)
            
            if not client_disconnected:
                elapsed_total = time.time() - start_time
                time_str = f"{int(elapsed_total // 60)}m {int(elapsed_total % 60)}s" if elapsed_total > 60 else f"{elapsed_total:.2f}s"
                try:
                    with open(temp_output_filename, "rb") as f: video_b64 = base64.b64encode(f.read()).decode('utf-8')
                    os.unlink(temp_output_filename)
                    yield json.dumps({ "status": "complete", "video_b64": video_b64, "file_name": video.filename, "elapsed_time": time_str, "total_objects": captured_count }) + "\n"
                except: yield json.dumps({"status": "error", "message": "Erro ao codificar vídeo final."}) + "\n"

        except Exception as e:
            logger.error(f"Erro no streaming: {e}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            if 'cap' in locals() and cap.isOpened(): cap.release()
            if 'out' in locals() and out.isOpened(): out.release()
            if os.path.exists(input_path):
                try: os.unlink(input_path)
                except: pass
            if client_disconnected and os.path.exists(temp_output_filename):
                try: os.unlink(temp_output_filename)
                except: pass

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global CAPTURE_HISTORY; CAPTURE_HISTORY = []
    config = { "conf_buraco": 0.25, "conf_bueiro": 0.30, "mode": "box", "resolution": 736, "pixel_cm2": 0.04, "measures": True, 
               "show_line": False, "show_id": True, "show_type": True, "show_conf": True, "show_dim": True, "show_area": True,
               "filter_size": False, "target_size": 0.90 }
    active_tracks = []; next_object_id = 1; visual_buffer = []; capture_seq_ref = [1]
    try:
        while True:
            data = await websocket.receive_text(); msg = json.loads(data)
            if msg['type'] == 'config': config.update(msg['data']); active_tracks = []; next_object_id = 1; CAPTURE_HISTORY = []; visual_buffer = []; capture_seq_ref = [1]
            elif msg['type'] == 'frame':
                frame_bytes = base64.b64decode(msg['data']); np_arr = np.frombuffer(frame_bytes, np.uint8); frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_det, _, active_tracks, next_object_id, new_thumbnails = processar_frame(
                        frame, config['conf_buraco'], config['conf_bueiro'], config['mode'], 
                        config['resolution'], config['measures'], config['pixel_cm2'] / 10.0, 
                        active_tracks, next_object_id, visual_buffer, use_roi=True,
                        show_trigger_line=config.get('show_line', False),
                        show_id=config.get('show_id', True), show_type=config.get('show_type', True), 
                        show_conf=config.get('show_conf', True), show_dim=config.get('show_dim', True), show_area=config.get('show_area', True),
                        capture_seq_ref=capture_seq_ref,
                        filter_by_size=config.get('filter_size', False), target_size=config.get('target_size', 0.90)
                    )
                    _, buffer = cv2.imencode('.jpg', frame_det, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    await websocket.send_json({ "image": base64.b64encode(buffer.tobytes()).decode('utf-8'), "thumbnails": new_thumbnails })
    except: print("WebSocket desconectado")
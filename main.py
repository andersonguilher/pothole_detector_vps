from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket, WebSocketDisconnect
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
import json
import time

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

# --- DIRETÓRIOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

print(f"📂 Diretório Base: {BASE_DIR}")

app.mount("/templates", StaticFiles(directory=TEMPLATES_DIR), name="templates_static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

models = {}
connected_monitors = [] 
CAPTURE_HISTORY = []

# --- CONFIGURAÇÃO PADRÃO ---
DEFAULT_CONFIG = {
    "conf_buraco": 0.25, "conf_bueiro": 0.30, "mode": "box", 
    "resolution": 640, "pixel_cm2": 0.04, "measures": True, 
    "show_id": True, "show_type": True, "show_conf": True, 
    "show_dim": True, "show_area": True, "filter_size": False, 
    "target_size": 0.90, "show_line": False, "rotation": 0
}

GLOBAL_CONFIG = DEFAULT_CONFIG.copy()

# --- PERSISTÊNCIA ---
def load_config():
    global GLOBAL_CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                GLOBAL_CONFIG.update(saved)
            # Força conversão para INT para evitar erros de string "270"
            if 'rotation' in GLOBAL_CONFIG:
                GLOBAL_CONFIG['rotation'] = int(GLOBAL_CONFIG['rotation'])
            print(f"💾 Config carregada! Rotação salva: {GLOBAL_CONFIG.get('rotation')}")
        except Exception as e: print(f"⚠️ Erro load config: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, 'w') as f: json.dump(GLOBAL_CONFIG, f, indent=4)
    except Exception as e: print(f"⚠️ Erro save config: {e}")

load_config()

# --- CARREGAR MODELOS ---
print("-" * 30)
def carregar_modelo(nome_arquivo, alias):
    caminho = os.path.join(MODELS_DIR, nome_arquivo)
    try:
        if not os.path.exists(caminho):
            print(f"❌ ARQUIVO FALTANDO: {caminho}")
            return False
        models[alias] = YOLO(caminho)
        print(f"✅ {alias.upper()} carregado.")
        return True
    except Exception as e:
        print(f"❌ ERRO {nome_arquivo}: {e}")
        return False

carregar_modelo("buracos_box.pt", "box")
carregar_modelo("buracos_seg.pt", "mask")
carregar_modelo("bueiros_det.pt", "bueiro")
print("-" * 30)

TRADUCAO_BUEIRO = { 'good': 'TAMPÃO BOM', 'broke': 'TAMPÃO QUEBRADO', 'uncover': 'BUEIRO ABERTO', 'missing': 'SEM TAMPA' }
CORES_MAPA = { 'BURACO': (0, 255, 255), 'TAMPÃO BOM': (0, 255, 0), 'TAMPÃO QUEBRADO': (0, 0, 255), 'BUEIRO ABERTO': (0, 0, 255), 'SEM TAMPA': (0, 0, 255) }

# --- FUNÇÕES AUXILIARES ---
def calcular_area_cm2(area_pixel, pixel_to_cm2): return area_pixel * pixel_to_cm2
def pixel_to_linear_measure(pixels, pixel_to_cm2): return pixels * math.sqrt(pixel_to_cm2)

def get_center_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2; cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2; cyB = (boxB[1] + boxB[3]) / 2
    return math.hypot(cxA - cxB, cyA - cyB)

def is_duplicate_capture(cx, cy, current_time, threshold_px=250, cooldown_sec=5.0):
    global CAPTURE_HISTORY
    CAPTURE_HISTORY = [c for c in CAPTURE_HISTORY if current_time - c['time'] < cooldown_sec]
    for c in CAPTURE_HISTORY:
        dist = math.hypot(cx - c['x'], cy - c['y'])
        if dist < threshold_px: return True
    return False

# Trava Visual (Histograma)
VISUAL_HISTORY = []
def is_visual_duplicate(img_crop, threshold=0.85):
    global VISUAL_HISTORY
    try:
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        for old_hist in VISUAL_HISTORY:
            if cv2.compareHist(hist, old_hist, cv2.HISTCMP_CORREL) > threshold: return True
        VISUAL_HISTORY.append(hist)
        if len(VISUAL_HISTORY) > 50: VISUAL_HISTORY.pop(0)
        return False
    except: return False

def desenhar_etiqueta(img, texto, x, y, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("arial.ttf", 14) 
    except: font = ImageFont.load_default()
    lines = texto.split('\n'); max_w = 0; total_h = 0; line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1] + 4
        max_w = max(max_w, w); total_h += h; line_heights.append(h)
    pad = 4; text_x = x; text_y = y - total_h - (pad * 2)
    if text_y < 0: text_y = y 
    bg_rgb = (color[2], color[1], color[0])
    text_color = (0, 0, 0) if (bg_rgb[0]*299 + bg_rgb[1]*587 + bg_rgb[2]*114)/1000 > 128 else (255, 255, 255)
    draw.rectangle([text_x, text_y, text_x + max_w + (pad*2), text_y + total_h + (pad*2)], fill=bg_rgb)
    curr_y = text_y + pad
    for i, line in enumerate(lines):
        draw.text((text_x + pad, curr_y), line, font=font, fill=text_color)
        curr_y += line_heights[i]
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- PROCESSAMENTO UNIFICADO ---
def processar_frame(img_original, config, active_tracks, next_object_id, visual_buffer, capture_seq_ref=None, processed_ids_set=None):
    global CAPTURE_HISTORY
    h_orig, w_orig = img_original.shape[:2]
    TRIGGER_Y = int(h_orig * 0.60)

    if config.get('show_line', False):
        cv2.line(img_original, (0, TRIGGER_Y), (w_orig, TRIGGER_Y), (0, 0, 255), 2)

    raw_detections = []
    generated_thumbnails = [] 
    
    # 1. Inferência
    if "bueiro" in models:
        res_b = models["bueiro"].predict(source=img_original, save=False, conf=0.01, iou=0.5, verbose=False, imgsz=config['resolution'])
        for box in res_b[0].boxes:
            conf = float(box.conf[0])
            if conf < config['conf_bueiro']: continue 
            label = TRADUCAO_BUEIRO.get(res_b[0].names[int(box.cls[0])], res_b[0].names[int(box.cls[0])])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            raw_detections.append({'box': [x1,y1,x2,y2], 'label': label, 'conf': conf, 'type': 'BUEIRO'})
    
    if models.get("mask" if config['mode'] == 'mask' else "box"):
        model = models["mask"] if config['mode'] == 'mask' else models["box"]
        res_p = model.predict(source=img_original, save=False, conf=0.01, iou=0.4, verbose=False, imgsz=config['resolution'])
        r = res_p[0]
        if hasattr(r, 'boxes'):
            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0])
                if conf < config['conf_buraco']: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if (x2-x1)*(y2-y1) < 300: continue
                cnt = None
                if config['mode'] == 'mask' and hasattr(r, 'masks') and r.masks is not None:
                    try: cnt = r.masks.xy[i].astype(np.int32)
                    except: pass
                raw_detections.append({'box': [x1,y1,x2,y2], 'label': 'BURACO', 'conf': conf, 'type': 'BURACO', 'cnt': cnt})

    # 2. Tracking
    new_active_tracks = []
    pixel_cm2 = config['pixel_cm2'] / 10.0
    for det in raw_detections:
        bx = det['box']
        w_m = pixel_to_linear_measure(bx[2]-bx[0], pixel_cm2)/100.0
        h_m = pixel_to_linear_measure(bx[3]-bx[1], pixel_cm2)/100.0
        if det['type'] == 'BUEIRO':
            real = max(w_m, h_m)
            if config.get('filter_size', False):
                tgt = config.get('target_size', 0.90)
                if not (tgt*0.96 <= real <= tgt*1.04): continue
            det['w_real'] = real; det['h_real'] = real
        else:
            det['w_real'] = w_m; det['h_real'] = h_m * 2.0 

        best_idx = -1; best_dist = float('inf')
        for i, track in enumerate(active_tracks):
            dist = get_center_distance(det['box'], track['box'])
            if det['label'] == track['label'] and dist < (w_orig * 0.15):
                if dist < best_dist: best_dist = dist; best_idx = i
        
        if best_idx != -1:
            track = active_tracks.pop(best_idx)
            track.update({'box': det['box'], 'conf': det['conf'], 'w_real': det['w_real'], 'h_real': det['h_real'], 'frames_missing': 0})
            if 'cnt' in det: track['cnt'] = det['cnt']
            new_active_tracks.append(track)
        else:
            new_track = {'id': next_object_id, 'box': det['box'], 'label': det['label'], 'type': det['type'], 'conf': det['conf'], 'w_real': det['w_real'], 'h_real': det['h_real'], 'frames_missing': 0, 'thumb_sent': False}
            if 'cnt' in det: new_track['cnt'] = det['cnt']
            new_active_tracks.append(new_track)
            next_object_id += 1

    for t in active_tracks:
        t['frames_missing'] += 1
        if t['frames_missing'] < 5: new_active_tracks.append(t)
    
    # 3. Captura
    curr_time = time.time()
    for track in new_active_tracks:
        if track['frames_missing'] > 0: continue
        x1,y1,x2,y2 = track['box']; cx = (x1+x2)//2; cy = (y1+y2)//2
        color = CORES_MAPA.get(track['label'], (255,0,255))
        
        if track.get('cnt') is not None: cv2.fillPoly(img_original, [track['cnt']], color)
        cv2.rectangle(img_original, (x1,y1), (x2,y2), color, 2)
        
        should_capture = False
        if processed_ids_set is not None and track['id'] in processed_ids_set: track['thumb_sent'] = True 
        
        if not track.get('thumb_sent', False):
            if cy > TRIGGER_Y: should_capture = True

        if should_capture:
            if not is_duplicate_capture(cx, cy, curr_time, 250, 5.0):
                margin = 10
                t_crop = img_original[max(0,y1-margin):min(h_orig,y2+margin), max(0,x1-margin):min(w_orig,x2+margin)]
                
                if t_crop.size > 0 and not is_visual_duplicate(t_crop):
                    target_w = 640; scale_f = target_w / w_orig; target_h = int(h_orig * scale_f)
                    frame_full = cv2.resize(img_original, (target_w, target_h))
                    _, buf = cv2.imencode('.jpg', frame_full, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    b64 = base64.b64encode(buf).decode('utf-8')
                    
                    display_id = track['id']
                    if capture_seq_ref: display_id = capture_seq_ref[0]; capture_seq_ref[0] += 1
                    
                    generated_thumbnails.append({
                        "id": display_id, "label": track['label'], "conf": track['conf'],
                        "dimensoes": f"{track['w_real']:.2f}m x {track['h_real']:.2f}m", "image": b64
                    })
                    track['thumb_sent'] = True
                    CAPTURE_HISTORY.append({'x': cx, 'y': cy, 'time': curr_time})
                    if processed_ids_set is not None: processed_ids_set.add(track['id'])

        txt = []
        if config['show_id']: txt.append(f"#{track['id']}")
        if config['show_type']: txt.append(track['label'])
        if config['show_conf']: txt.append(f"{track['conf']:.2f}")
        label = " ".join(txt)
        if config['measures'] and config['show_dim']: label += f"\n{track['w_real']:.2f}m x {track['h_real']:.2f}m"
        desenhar_etiqueta(img_original, label, x1, y1, color)

    return img_original, new_active_tracks, next_object_id, generated_thumbnails

# --- WEBSOCKETS ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse("index.html", {"request": request})
@app.get("/live", response_class=HTMLResponse)
async def mobile_page(request: Request): return templates.TemplateResponse("mobile.html", {"request": request})
@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request): return templates.TemplateResponse("monitor.html", {"request": request})

@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    global GLOBAL_CONFIG
    await websocket.accept()
    await websocket.send_json({'type': 'config_update', 'data': GLOBAL_CONFIG})
    connected_monitors.append(websocket)
    try:
        while True:
            data = await websocket.receive_text(); msg = json.loads(data)
            if msg.get('type') == 'config':
                GLOBAL_CONFIG.update(msg['data']); save_config()
            elif msg.get('type') == 'reset':
                GLOBAL_CONFIG = DEFAULT_CONFIG.copy(); save_config()
                global CAPTURE_HISTORY, VISUAL_HISTORY; CAPTURE_HISTORY = []; VISUAL_HISTORY = []
                for mon in connected_monitors: await mon.send_json({'type': 'config_update', 'data': GLOBAL_CONFIG})
    except:
        if websocket in connected_monitors: connected_monitors.remove(websocket)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_tracks = []; next_object_id = 1; visual_buffer = []
    processed_ids_set = set()
    global VISUAL_HISTORY; VISUAL_HISTORY = [] 
    frame_count = 0
    
    try:
        while True:
            data = await websocket.receive_text(); msg = json.loads(data)
            
            if msg['type'] == 'frame':
                frame_bytes = base64.b64decode(msg['data'])
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # BLINDAGEM DE ROTAÇÃO
                    rot = int(GLOBAL_CONFIG.get('rotation', 0))
                    
                    if rot == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rot == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    frame_det, active_tracks, next_object_id, new_thumbs = processar_frame(
                        frame, GLOBAL_CONFIG, active_tracks, next_object_id, visual_buffer, 
                        capture_seq_ref=None, processed_ids_set=processed_ids_set
                    )
                    
                    # LOG DE DEBUG A CADA 60 FRAMES
                    frame_count += 1
                    if frame_count % 60 == 0:
                        print(f"[STATUS] Frame OK | Rotação Ativa: {rot} | Resolução: {GLOBAL_CONFIG['resolution']}")

                    _, buffer = cv2.imencode('.jpg', frame_det, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    img_mon_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    
                    # Contra-Rotação
                    frame_mobile = frame_det.copy()
                    if rot == 90: frame_mobile = cv2.rotate(frame_mobile, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rot == 180: frame_mobile = cv2.rotate(frame_mobile, cv2.ROTATE_180)
                    elif rot == 270: frame_mobile = cv2.rotate(frame_mobile, cv2.ROTATE_90_CLOCKWISE)
                    _, buf_mob = cv2.imencode('.jpg', frame_mobile, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    img_mob_b64 = base64.b64encode(buf_mob.tobytes()).decode('utf-8')
                    
                    await websocket.send_json({ "image": img_mob_b64, "thumbnails": new_thumbs })
                    
                    if connected_monitors:
                        payload = { "image": img_mon_b64, "detections": new_thumbs }
                        for mon in reversed(connected_monitors):
                            try: await mon.send_json(payload)
                            except: connected_monitors.remove(mon)
            
            elif msg['type'] == 'config':
                # Protege a rotação contra overwrite do mobile
                incoming_data = msg['data']
                if 'rotation' not in incoming_data and 'rotation' in GLOBAL_CONFIG:
                    incoming_data['rotation'] = GLOBAL_CONFIG['rotation']
                
                GLOBAL_CONFIG.update(incoming_data)
                save_config()

    except WebSocketDisconnect: print("Mobile desconectado")
    except Exception as e: print(f"Erro WS: {e}")

@app.post("/detectar_video")
async def detectar_video(
    request: Request, video: UploadFile = File(...), 
    conf_buraco: float = Form(0.25), conf_bueiro: float = Form(0.30), 
    detection_mode: str = Form("box"), resolution_ia: int = Form(640), 
    pixel_to_cm2: float = Form(0.04), calculate_measures: str = Form("true"), 
    show_trigger_line: bool = Form(False), show_id: bool = Form(True), 
    show_type: bool = Form(True), show_conf: bool = Form(True), 
    show_dim: bool = Form(True), show_area: bool = Form(True),
    filter_by_size: str = Form("false"), target_size: float = Form(0.90)
):
    try:
        video_config = {
            "conf_buraco": conf_buraco, "conf_bueiro": conf_bueiro, "mode": detection_mode,
            "resolution": resolution_ia, "pixel_cm2": pixel_to_cm2, "measures": (calculate_measures.lower()=='true'),
            "show_line": show_trigger_line, "show_id": show_id, "show_type": show_type,
            "show_conf": show_conf, "show_dim": show_dim, "show_area": show_area,
            "filter_size": (filter_by_size.lower()=='true'), "target_size": target_size
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.split('.')[-1]}") as tmp:
            tmp.write(await video.read())
            input_path = tmp.name

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        async def process_stream():
            active_tracks = []; next_object_id = 1; capture_seq_ref = [1]; visual_buffer = []
            frame_current = 0
            # IMPORTANTE: Reseta históricos para o vídeo
            global VISUAL_HISTORY, CAPTURE_HISTORY
            VISUAL_HISTORY = []; CAPTURE_HISTORY = []
            
            try:
                while cap.isOpened():
                    if await request.is_disconnected(): break
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_det, active_tracks, next_object_id, new_thumbnails = processar_frame(
                        frame, video_config, active_tracks, next_object_id, visual_buffer, 
                        capture_seq_ref=capture_seq_ref, processed_ids_set=None
                    )
                    
                    frame_current += 1
                    if frame_current % 3 == 0:
                        _, buffer = cv2.imencode('.jpg', cv2.resize(frame_det, (640, 360)), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        yield json.dumps({ 
                            "status": "processing", "current": frame_current, "total": total_frames, 
                            "objs": capture_seq_ref[0]-1, "preview": base64.b64encode(buffer.tobytes()).decode('utf-8'), 
                            "thumbnails": new_thumbnails 
                        }) + "\n"
                
                yield json.dumps({ "status": "complete", "elapsed_time": "Finalizado", "file_name": video.filename }) + "\n"
            finally:
                cap.release()
                os.unlink(input_path)

        return StreamingResponse(process_stream(), media_type="application/x-ndjson")
    
    except Exception as e: return JSONResponse({"message": f"Erro: {e}"}, 500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
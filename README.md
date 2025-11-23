# 🕳️ Pothole Detector – Documentação Completa

Aplicação Python FastAPI para detectar buracos em ruas utilizando YOLOv8 e exibir resultados com overlay visual.

---

## 📦 Estrutura do Projeto

```
📂 detector/
 ├── main.py
 ├── templates/
 │    └── upload.html
 ├── static/
 │    └── style.css (opcional)
 └── requirements.txt
```

---

## 🧠 Modelo

- Usa YOLOv8 (`ultralytics`)
- Aceita imagem enviada via formulário
- Retorna imagem processada com caixas desenhadas

---

## 🔧 Requisitos

```
fastapi
uvicorn
jinja2
python-multipart
ultralytics
opencv-python
numpy
```

---

## 🚀 Como Executar

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Acesse em:

```
http://SEU_SERVIDOR/detector/
```

---

## 🔀 Configuração do Proxy Reverso (Nginx)

```nginx
location /detector/ {
    proxy_pass http://127.0.0.1:8000/;  # A barra no final é importante
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Suporte a WebSocket (futuro)
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

---

## 📝 Observações

- A barra no final do `proxy_pass` é obrigatória para manter rotas corretas.
- Ideal para uso junto com sistemas de georreferenciamento ou dashboards urbanos.

---

## 📄 Licença

Livre para uso e modificação.

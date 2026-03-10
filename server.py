from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Carregamento dos modelos (Apenas uma vez no início)
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def processar_e_alinhar(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        margem = int(w * 0.5)
        y1, y2 = max(0, y - margem), min(img.shape[0], y + h + margem)
        x1, x2 = max(0, x - int(margem / 2)), min(img.shape[1], x + w + int(margem / 2))
        img_crop = img[y1:y2, x1:x2]
    else:
        img_crop = img

    img_res = cv2.resize(img_crop, (600, 800))
    _, buffer = cv2.imencode('.jpg', img_res)
    return img_res, base64.b64encode(buffer).decode('utf-8')

def extrair_embedding(img_cv2):
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    face_tensor = mtcnn(img_pil)
    if face_tensor is None: return None
    with torch.no_grad():
        return resnet(face_tensor.unsqueeze(0)).cpu().numpy().flatten()

@app.route('/extrair', methods=['POST'])
def extrair():
    if 'foto' not in request.files:
        return jsonify({"erro": "Sem foto"}), 400
    
    img_proc, foto_b64 = processar_e_alinhar(request.files['foto'].read())
    if img_proc is None: return jsonify({"erro": "Erro processamento"}), 400
    
    embedding = extrair_embedding(img_proc)
    if embedding is None:
        return jsonify({"sucesso": False, "erro": "Face não detectada", "foto_b64": foto_b64}), 422

    return jsonify({
        "sucesso": True,
        "foto_b64": foto_b64,
        "embedding": embedding.tolist() # Retorna como lista para o PHP
    })

@app.route('/comparar', methods=['POST'])
def comparar():
    data = request.json
    emb_novo = np.array(data['novo'], dtype=np.float32)
    lista_conhecidos = np.array(data['conhecidos'], dtype=np.float32)
    
    if len(lista_conhecidos) == 0:
        return jsonify({"max_sim": 0})

    sims = cosine_similarity([emb_novo], lista_conhecidos)[0]
    return jsonify({"max_sim": float(np.max(sims)), "index": int(np.argmax(sims))})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
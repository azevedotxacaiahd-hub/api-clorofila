from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ==================================================
# FUNÇÕES DO MODELO
# ==================================================

def segmentar_folha(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        maior_contorno = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [maior_contorno], -1, 255, -1)
        folha_segmentada = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(maior_contorno)
        folha_recortada = folha_segmentada[y:y+h, x:x+w]
        return folha_recortada
    else:
        return img

def extract_texture_features(gray_img):
    features = []
    if gray_img.shape[0] > 200:
        gray_img = cv2.resize(gray_img, (200, 200))
    
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    features.append(np.mean(gray_img))
    features.append(np.std(gray_img))
    features.append(np.percentile(gray_img, 25))
    features.append(np.percentile(gray_img, 50))
    features.append(np.percentile(gray_img, 75))
    
    hist_nonzero = hist[hist > 0]
    if len(hist_nonzero) > 0:
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    else:
        entropy = 0
    features.append(entropy)
    
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    local_var = (gray_img.astype(np.float32) - local_mean) ** 2
    local_var = cv2.filter2D(local_var, -1, kernel)
    features.append(np.mean(local_var))
    features.append(np.std(local_var))
    features.append(np.max(local_var))
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    features.append(np.mean(grad_mag))
    features.append(np.std(grad_mag))
    features.append(np.max(grad_mag))
    
    return np.array(features)

def extract_all_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    img_segmentada = segmentar_folha(img)
    if img_segmentada is None or img_segmentada.shape[0] == 0:
        img_segmentada = img
    img_segmentada = cv2.resize(img_segmentada, (256, 256))
    
    img_rgb = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2LAB)
    
    features = []
    
    for i in range(3):
        features.append(np.mean(img_rgb[:,:,i]))
        features.append(np.std(img_rgb[:,:,i]))
    
    for i in range(3):
        features.append(np.mean(img_hsv[:,:,i]))
        features.append(np.std(img_hsv[:,:,i]))
    
    for i in range(3):
        features.append(np.mean(img_lab[:,:,i]))
        features.append(np.std(img_lab[:,:,i]))
    
    R = img_rgb[:,:,0].astype(np.float32)
    G = img_rgb[:,:,1].astype(np.float32)
    B = img_rgb[:,:,2].astype(np.float32)
    
    gli = (2*G - R - B) / (2*G + R + B + 1e-6)
    features.append(np.mean(gli))
    features.append(np.std(gli))
    ndgi = (G - R) / (G + R + 1e-6)
    features.append(np.mean(ndgi))
    features.append(np.mean(G / (R + 1e-6)))
    features.append(np.mean(R / (G + 1e-6)))
    features.append(np.mean((G - B) / (G + B + 1e-6)))
    
    gray = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2GRAY)
    texture_features = extract_texture_features(gray)
    features.extend(texture_features)
    
    return np.array(features)

# ==================================================
# CARREGAR MODELO
# ==================================================

# Baixar modelos do GitHub (ou usar local)
import requests
import os

# Se os modelos não existirem, baixar do repositório
if not os.path.exists('modelo_xgboost.pkl'):
    print("Baixando modelo...")
    # Você precisa fazer upload dos seus modelos para o Render via arquivo

model = joblib.load('modelo_xgboost.pkl')
scaler = joblib.load('scaler_xgboost.pkl')
print("✅ Modelo carregado com sucesso!")

# ==================================================
# ROTAS DA API
# ==================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'API para análise de Clorofila a, b e Carotenoides',
        'endpoints': {
            '/health': 'GET - Verifica o status da API',
            '/predict': 'POST - Envie uma imagem para análise'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'modelo_carregado': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Arquivo vazio'}), 400
        
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        features = extract_all_features(temp_path)
        
        if features is None:
            os.remove(temp_path)
            return jsonify({'error': 'Erro ao processar imagem'}), 400
        
        features_scaled = scaler.transform([features])
        chl_a, chl_b, car = model.predict(features_scaled)[0]
        
        ratio = chl_a / chl_b if chl_b > 0 else 0
        
        if chl_a > 40:
            interpretacao = "Folha muito saudável"
            status = "saudavel"
        elif chl_a > 25:
            interpretacao = "Folha saudável (teor moderado)"
            status = "moderado"
        else:
            interpretacao = "Possível deficiência nutricional"
            status = "deficiencia"
        
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'resultados': {
                'clorofila_a': round(float(chl_a), 2),
                'clorofila_b': round(float(chl_b), 2),
                'carotenoides': round(float(car), 2),
                'razao_chl_a_b': round(float(ratio), 2),
                'interpretacao': interpretacao,
                'status': status
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
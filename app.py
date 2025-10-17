import os, json, joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY', 'SWE_DERIV_2025')
PORT = int(os.getenv('PORT', '10000'))

app = Flask(__name__)

MODEL_DIR = 'model'
DATA_DIR = 'data'
INCOMING = os.path.join(DATA_DIR, 'incoming')
PROCESSED = os.path.join(DATA_DIR, 'processed')
SNAPSHOTS = os.path.join(DATA_DIR, 'snapshots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INCOMING, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(SNAPSHOTS, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_rf.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'columns.json')

DEFAULT_COLUMNS = ['trend_len','rango','vel','vol20','slope10','imb20','rev_z','rsi7']

def _auth_ok(req):
    return req.headers.get('X-API-KEY') == API_KEY

def _load_model():
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(COLUMNS_PATH)):
        return None, DEFAULT_COLUMNS
    try:
        model = joblib.load(MODEL_PATH)
        with open(COLUMNS_PATH, 'r', encoding='utf-8') as f:
            cols = json.load(f)
        return model, cols
    except Exception as e:
        print(f'[load_model] Error: {e}')
        return None, DEFAULT_COLUMNS

model, model_cols = _load_model()
print(f'✅ Modelo cargado: {MODEL_PATH if model else "BASELINE (0.5)"}')

def _predict_score(payload: dict):
    x = [[float(payload.get(c, 0)) for c in model_cols]]
    score = 0.5 if model is None else float(model.predict_proba(np.array(x))[0, 1])
    threshold = 0.65
    decision = 'ENTER' if score >= threshold else 'NO_ENTER'
    return score, threshold, decision

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'ok': True,
        'model_loaded': model is not None,
        'columns': model_cols,
        'time': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/evaluar_entrada', methods=['POST'])
def evaluar_entrada():
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json(force=True) or {}
    score, thr, decision = _predict_score(data)
    return jsonify({
        'score': score,
        'threshold': thr,
        'decision': decision,
        'explanation': 'Dummy model until training data available.'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

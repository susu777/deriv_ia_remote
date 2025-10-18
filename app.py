from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# =====================================================
# 🔹 Cargar modelo entrenado IA
# =====================================================
MODEL_PATH = os.path.join(os.getcwd(), "train", "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo IA cargado correctamente desde:", MODEL_PATH)
except Exception as e:
    print("⚠️ Error al cargar modelo:", e)
    model = None


# =====================================================
# 🔹 Ruta base: verificación
# =====================================================
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "🚀 Servidor IA remoto operativo para Deriv V75 Bot."
    })


# =====================================================
# 🔹 Endpoint principal: recibir datos y devolver predicción
# =====================================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    try:
        data = request.get_json()

        # ============================================
        # 🔸 Espera datos como los del CSV del bot:
        # ============================================
        # {
        #   "trend_len": 8,
        #   "rango": 0.0021,
        #   "velocidad": 0.00034,
        #   "vol20": 0.0018,
        #   "slope10": 0.0006,
        #   "imb20": 0.42,
        #   "rev_z": -0.23,
        #   "rsi7": 65.2,
        #   "bb_pos": 0.78
        # }
        # ============================================

        features = [
            data["trend_len"],
            data["rango"],
            data["velocidad"],
            data["vol20"],
            data["slope10"],
            data["imb20"],
            data["rev_z"],
            data["rsi7"],
            data["bb_pos"]
        ]

        X = np.array(features).reshape(1, -1)
        proba = model.predict_proba(X)[0][1]  # probabilidad de ganar

        decision = "CALL" if proba >= 0.6 else "NO-TRADE"

        return jsonify({
            "probabilidad": round(float(proba), 4),
            "decision": decision,
            "umbral_usado": 0.6,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

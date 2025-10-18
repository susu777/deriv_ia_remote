from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# =====================================
# 🔹 Inicializar la app Flask
# =====================================
app = Flask(__name__)

# =====================================
# 🔹 Cargar modelo IA entrenado
# =====================================
MODEL_PATH = os.path.join(os.getcwd(), "train", "model.pkl")  # Ajusta si tu modelo está en otra ruta
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print("⚠️ Error al cargar el modelo:", e)
    model = None


# =====================================
# 🔹 Ruta raíz para verificar estado
# =====================================
@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "🚀 Servidor IA remoto activo y funcionando correctamente en Render."
    })


# =====================================
# 🔹 Endpoint de predicción
# =====================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    try:
        data = request.get_json()

        # Espera datos como: {"features": [valor1, valor2, valor3, ...]}
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400


# =====================================
# 🔹 Iniciar servidor
# =====================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

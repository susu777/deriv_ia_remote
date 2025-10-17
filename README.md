# 🧠 Deriv IA Remota

Servidor Flask para predicción y aprendizaje en tiempo real conectado con tu bot Deriv.

## 🚀 Despliegue en Render

1. Sube esta carpeta a un repositorio en GitHub.
2. En Render.com, crea un **New Web Service**:
   - **Build command:** pip install -r requirements.txt
   - **Start command:** python app.py
   - **Environment Variables:**
     - API_KEY=SWE_DERIV_2025
     - PORT=10000
3. Una vez activo, tu API estará disponible en:
   \https://deriv-ia-swe.onrender.com\

## 📡 Endpoints

- **GET /status** → estado del modelo
- **POST /evaluar_entrada** → evalúa una posible entrada
- **POST /ingestar_operacion** → guarda resultados para reentrenamiento
- **POST /reload_model** → recarga modelo tras entrenamiento


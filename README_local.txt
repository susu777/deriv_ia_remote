# 🧠 Prueba local del servidor IA

## 1️⃣ Instalar dependencias
pip install -r requirements.txt

## 2️⃣ Ejecutar servidor local
python app.py

## 3️⃣ Probar estado
Abrir en navegador:
http://127.0.0.1:10000/status

## 4️⃣ Ejemplo de prueba con curl:
curl -X POST http://127.0.0.1:10000/evaluar_entrada -H "Content-Type: application/json" -H "X-API-KEY: SWE_DERIV_2025" -d "{\"trend_len\":6,\"rango\":0.0011,\"vel\":0.0004,\"vol20\":0.002,\"slope10\":0.0001,\"imb20\":0.2,\"rev_z\":0.9,\"rsi7\":49}"


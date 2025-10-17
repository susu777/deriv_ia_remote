import os, json, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import unify_incoming

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, 'model')
DATA_DIR  = os.path.join(BASE, 'data')
INCOMING  = os.path.join(DATA_DIR, 'incoming')
PROCESSED = os.path.join(DATA_DIR, 'processed')

MODEL_PATH   = os.path.join(MODEL_DIR, 'modelo_rf.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'columns.json')

COLS = ['trend_len','rango','vel','vol20','slope10','imb20','rev_z','rsi7']

def main():
    csv_path = unify_incoming(INCOMING, PROCESSED)
    if not csv_path:
        print('No hay datos nuevos para entrenar.')
        return
    df = pd.read_csv(csv_path).dropna(subset=COLS + ['label'])
    X, y = df[COLS].astype(float), df['label'].astype(int)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y)
    print(f'Modelo entrenado. Precisión: {acc:.3f} con {len(X)} filas.')
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(COLUMNS_PATH, 'w', encoding='utf-8') as f:
        json.dump(COLS, f, ensure_ascii=False)
    print('Modelo guardado.')
if __name__ == '__main__':
    main()

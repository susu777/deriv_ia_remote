import os, json, glob, shutil
import pandas as pd
from datetime import datetime

def unify_incoming(incoming_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    rows = []
    for fp in glob.glob(os.path.join(incoming_dir, '*.json')):
        with open(fp, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        feats = obj.get('features', {})
        row = {
            'fecha_hora': obj.get('fecha_hora'),
            'direccion': obj.get('direccion'),
            'resultado': obj.get('resultado'),
            'profit': obj.get('profit', 0.0),
            'symbol': obj.get('symbol', '')
        }
        for k,v in feats.items(): row[k] = v
        rows.append(row)
    if not rows: return None
    df = pd.DataFrame(rows)
    df['label'] = (df['profit'] > 0).astype(int)
    df.dropna(subset=['trend_len','rango','vel','vol20','slope10','imb20','rev_z','rsi7','label'], inplace=True)
    out_csv = os.path.join(processed_dir, 'dataset_unificado.csv')
    df.to_csv(out_csv, index=False)
    return out_csv

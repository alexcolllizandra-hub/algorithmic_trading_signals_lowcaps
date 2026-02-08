"""Script para ejecutar el pipeline completo."""

import sys
sys.path.insert(0, 'backend')
from pathlib import Path

DATA_PATH = Path('data')
MODEL_PATH = DATA_PATH / 'processed' / 'models'

print('=== PIPELINE TRADING MVP ===')
print()

# 1. Screener
print('[1/5] Ejecutando screener...')
from screener import run_screener
symbols = run_screener(DATA_PATH, count=8)
print(f'Tickers obtenidos: {symbols}')
print()

# 2. Market Data
print('[2/5] Descargando datos historicos...')
from market_data import download_all_tickers
all_data = download_all_tickers(symbols, DATA_PATH)
print(f'Tickers descargados: {len(all_data)}')
print()

# 3. Features
print('[3/5] Generando features con Polars LazyFrame...')
from features import engineer_features, save_training_data, split_temporal
df = engineer_features(DATA_PATH)
save_training_data(df, DATA_PATH)
print(f'Features generadas: {df.shape}')
print()

# 4. Model
print('[4/5] Entrenando modelo...')
from model import train_model, save_model
train_df, test_df = split_temporal(df)
model_info = train_model(train_df)
save_model(model_info, MODEL_PATH)
print(f'Modelo: {model_info["model_name"]} (AUC={model_info["auc"]:.4f})')
print()

# 5. Signals
print('[5/5] Generando senales...')
from model import predict
from signals import generate_signals, get_latest_signals, get_best_opportunity
df = predict(model_info, df)
df = generate_signals(df)
signals = get_latest_signals(df)

print()
print('=== RESULTADOS ===')
print(f'Total senales: {len(signals)}')
for s in signals[:5]:
    print(f'  {s["symbol"]}: {s["signal"]} (prob={s["probability"]:.2f})')

best = get_best_opportunity(signals)
if best:
    print()
    print(f'MEJOR OPORTUNIDAD: {best["symbol"]} - {best["signal"]} (prob={best["probability"]:.2f})')

print()
print('=== PIPELINE COMPLETADO ===')

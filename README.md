# Trading Algoritmico MVP

Sistema de trading algoritmico academico que genera señales de compra/venta usando machine learning.

**Autores:** Alex Coll, Emilio Delgado  
**Asignatura:** MD007 - Estructura de Datos

## Que hace el proyecto

1. **Web Scraping** - obtiene tickers de small-cap stocks via API de Yahoo Finance
2. **Descarga de datos** - historicos OHLCV de 1 año con yfinance
3. **EDA** - analisis exploratorio con Polars (lazy evaluation)
4. **Feature Engineering** - retornos, volatilidad, momentum, volumen relativo
5. **Modelado** - entrena GradientBoosting, RandomForest, LogisticRegression con TimeSeriesSplit
6. **Señales** - genera BUY/SELL/HOLD segun probabilidades del modelo
7. **Backtesting** - evalua estrategia con reglas TP/SL (5%/3%)
8. **Visualizacion** - dashboard HTML interactivo con Chart.js

## Estructura del proyecto

```
├── src/                    # codigo principal
│   ├── main.py            # orquestador del pipeline
│   ├── screener.py        # web scraping yahoo finance api
│   ├── market_data.py     # descarga datos con yfinance
│   ├── eda.py             # analisis exploratorio
│   ├── features.py        # feature engineering con polars
│   ├── model.py           # entrenamiento sklearn
│   ├── signals.py         # generacion de señales
│   ├── backtest.py        # backtesting con tp/sl
│   ├── visualization.py   # generador de dashboard html
│   └── database.py        # persistencia mongodb (opcional)
├── frontend/              # react app (opcional)
├── data/                  # datos generados (se crea automaticamente)
│   ├── raw/              # screener.csv, market_data.parquet
│   └── processed/        # features.parquet, signals.csv, models/
├── dashboard.html         # dashboard generado
├── backtest_report.json   # resultados del backtest
├── requirements.txt       # dependencias
└── run_pipeline.py        # script de ejecucion alternativo
```

## Instalacion

```bash
# clonar repositorio
git clone <url-del-repo>
cd MD007-S4-Alex_Coll_Emilio_Delgado

# crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate  # windows
source venv/bin/activate  # linux/mac

# instalar dependencias
pip install -r requirements.txt
```

## Ejecucion

```bash
# ejecutar pipeline completo
cd src
python main.py

# opciones disponibles
python main.py --skip-screener    # usa screener guardado
python main.py --skip-training    # usa modelo guardado
```

El pipeline genera:
- `data/raw/screener.csv` - tickers obtenidos
- `data/raw/market_data.parquet` - datos historicos
- `data/processed/features.parquet` - features calculadas
- `data/processed/models/sklearn_model.pkl` - modelo entrenado
- `backtest_report.json` - metricas del backtest
- `dashboard.html` - visualizacion interactiva

## Tecnologias

| Componente | Tecnologia |
|------------|------------|
| Web Scraping | requests + Yahoo Finance API |
| Procesamiento | Polars (Lazy Evaluation) |
| Modelado | scikit-learn (TimeSeriesSplit) |
| Visualizacion | Chart.js + HTML |
| Base de datos | MongoDB (opcional) |
| Frontend | React + Tailwind (opcional) |

## Variable objetivo

```
target = 1 si el precio sube >0% en los proximos 5 dias
target = 0 en caso contrario
```

Clasificacion binaria evaluada con AUC-ROC.

## Logica de señales

| Señal | Condicion |
|-------|-----------|
| STRONG_BUY | probabilidad > 70% AND precio > SMA_50 |
| BUY | probabilidad > 60% |
| SELL | probabilidad < 40% |
| HOLD | probabilidad entre 40-60% |

## Backtesting

Simula trades con reglas de gestion de riesgo:
- Take Profit: 5%
- Stop Loss: 3%
- Horizonte maximo: 10 dias

Metricas calculadas: Return Total, Win Rate, Max Drawdown, Sharpe Ratio.

## Requisitos del sistema

- Python 3.9+
- Conexion a internet (para descargar datos)
- MongoDB (opcional, para persistencia)

## Dependencias principales

```
polars>=0.20.0
yfinance>=0.2.0
scikit-learn>=1.3.0
requests>=2.31.0
numpy>=1.24.0
```

Ver `requirements.txt` para lista completa.

## Notas

- el modelo predice probabilidades, no retornos exactos
- los mercados son impredecibles, AUC cercano a 0.50-0.55 es normal
- proyecto academico, no usar para trading real
- TimeSeriesSplit evita data leakage en la validacion

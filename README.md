# Trading Algoritmico MVP

Sistema de trading algoritmico academico que genera señales de compra/venta para acciones small-cap usando machine learning.

**Autores:** Alex Coll, Emilio Delgado  
**Asignatura:** MD007 - Estructura de Datos y su Almacenamiento  
**Fecha:** Febrero 2026

## Que hace el proyecto

1. **Web Scraping** - obtiene tickers de small-cap stocks via API JSON de Yahoo Finance
2. **Descarga de datos** - historicos OHLCV de 1 año con yfinance
3. **EDA** - analisis exploratorio con Polars (lazy evaluation)
4. **Feature Engineering** - retornos, volatilidad, momentum, volumen relativo
5. **Modelado** - entrena GradientBoosting, RandomForest, LogisticRegression con TimeSeriesSplit
6. **Señales** - genera BUY/SELL/HOLD segun probabilidades del modelo
7. **Backtesting** - evalua estrategia con reglas TP/SL (5%/3%)
8. **Persistencia** - almacena resultados en MongoDB
9. **API REST** - expone datos via FastAPI
10. **Visualizacion** - dashboard HTML interactivo + frontend React

## Arquitectura por capas

```
src/
├── main.py                    # orquestador del pipeline
│
├── ingestion/                 # CAPA DE INGESTION
│   ├── screener.py           # web scraping yahoo finance api
│   └── market_data.py        # descarga historicos yfinance
│
├── processing/                # CAPA DE PROCESAMIENTO
│   ├── eda.py                # analisis exploratorio, limpieza
│   └── features.py           # feature engineering con polars lazy
│
├── modeling/                  # CAPA DE MODELADO
│   ├── model.py              # entrenamiento sklearn + timeseriessplit
│   ├── signals.py            # generacion de señales BUY/SELL/HOLD
│   └── backtest.py           # backtesting con tp/sl
│
├── persistence/               # CAPA DE PERSISTENCIA
│   └── database.py           # mongodb (singleton pattern)
│
└── presentation/              # CAPA DE PRESENTACION
    └── visualization.py      # generador de dashboard html

api/                           # API REST
├── main.py                   # fastapi app
└── routes/                   # endpoints
    ├── market.py             # GET /api/tickers, /api/market/{ticker}
    ├── signals.py            # GET /api/signals/{ticker}
    ├── backtest.py           # GET /api/backtest, /api/backtest/kpis
    ├── pipeline.py           # POST /api/pipeline/run
    └── dashboard.py          # GET /api/dashboard

frontend/                      # REACT APP
├── src/
│   ├── App.jsx               # componente principal
│   └── components/           # SignalCard, PriceChart, etc.
└── package.json
```

## Instalacion

```bash
# clonar repositorio
git clone https://github.com/alexcolllizandra-hub/algorithmic_trading_signals_lowcaps.git
cd algorithmic_trading_signals_lowcaps

# crear entorno virtual
python -m venv venv
venv\Scripts\activate  # windows
source venv/bin/activate  # linux/mac

# instalar dependencias python
pip install -r requirements.txt

# instalar dependencias frontend (opcional)
cd frontend
npm install
```

## Ejecucion

### Pipeline completo

```bash
# desde raiz del proyecto
python run_pipeline.py

# o desde src/
cd src
python main.py

# opciones
python main.py --skip-screener    # usa screener guardado
python main.py --skip-training    # usa modelo guardado
```

### API REST

```bash
# lanzar api en puerto 8000
python run_api.py

# endpoints disponibles en http://localhost:8000/docs
```

### Frontend React

```bash
cd frontend
npm run dev

# disponible en http://localhost:5173
```

### Dashboard estatico

Abrir `dashboard.html` en el navegador tras ejecutar el pipeline.

## Archivos generados

```
data/
├── raw/
│   ├── screener.csv           # tickers obtenidos
│   └── market_data.parquet    # datos OHLCV
└── processed/
    ├── features.parquet       # features calculadas
    ├── signals.csv            # señales generadas
    ├── eda_report.json        # reporte EDA
    └── models/
        └── sklearn_model.pkl  # modelo entrenado

backtest_report.json           # metricas del backtest
dashboard.html                 # visualizacion interactiva
```

## Stack tecnologico

| Capa | Tecnologia | Descripcion |
|------|------------|-------------|
| Ingestion | requests, yfinance | APIs de Yahoo Finance |
| Procesamiento | Polars | DataFrames con Lazy Evaluation |
| Modelado | scikit-learn | GradientBoosting, RandomForest, LogisticRegression |
| Validacion | TimeSeriesSplit | Evita data leakage en series temporales |
| Persistencia | MongoDB, Parquet | NoSQL + formato columnar |
| API | FastAPI, Uvicorn | Framework asincrono + servidor ASGI |
| Frontend | React, Vite, Tailwind | SPA moderna |
| Visualizacion | Chart.js | Graficos interactivos |

## Variable objetivo

```
target = 1 si el precio sube >0% en los proximos 5 dias
target = 0 en caso contrario
```

Clasificacion binaria evaluada con AUC-ROC usando TimeSeriesSplit (5 folds).

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

Metricas calculadas:
- Return Total (P&L acumulado)
- Win Rate (% trades ganadores)
- Max Drawdown (mayor caida desde maximo)
- Sharpe Ratio (rentabilidad ajustada por riesgo)

## Patrones de diseño

| Patron | Implementacion | Proposito |
|--------|----------------|-----------|
| Singleton | MongoDBConnection | Una unica conexion a BD |
| Builder/Fluent | DataCleaner | Operaciones encadenables |
| Strategy | screener fallbacks | Multiples fuentes de datos |
| Factory | ModelTrainer._get_models() | Crear modelos dinamicamente |
| Repository | database.py | Abstraccion de persistencia |

## API Endpoints

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | /api/tickers | Lista de tickers disponibles |
| GET | /api/market/{ticker} | Datos OHLCV |
| GET | /api/signals/{ticker} | Señales de un ticker |
| GET | /api/backtest/kpis | KPIs principales |
| GET | /api/dashboard | Datos agregados para frontend |
| POST | /api/pipeline/run | Ejecutar pipeline |

Documentacion Swagger: http://localhost:8000/docs

## Requisitos del sistema

- Python 3.9+
- Node.js 18+ (para frontend)
- MongoDB 6+ (opcional, para persistencia)
- Conexion a internet (para descargar datos)

## Notas

- el modelo predice probabilidades, no retornos exactos
- los mercados son impredecibles, AUC cercano a 0.50-0.55 es normal
- proyecto academico, no usar para trading real
- TimeSeriesSplit evita data leakage en la validacion
- si el puerto 8000 esta ocupado, matar proceso con: `Get-NetTCPConnection -LocalPort 8000 | Stop-Process`

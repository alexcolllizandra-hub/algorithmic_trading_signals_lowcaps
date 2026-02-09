"""
visualizacion web del sistema de trading
genera dashboard html interactivo con chart.js
"""

import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> pl.DataFrame:
    for path in [data_path / "processed" / "training_data.parquet",
                 data_path / "processed" / "features.parquet",
                 data_path / "raw" / "market_data.parquet"]:
        if path.exists():
            return pl.read_parquet(path)
    raise FileNotFoundError(f"no hay datos en: {data_path}")


def load_model(data_path: Path) -> Optional[Dict]:
    for name in ["model.pkl", "sklearn_model.pkl"]:
        path = data_path / "processed" / "models" / name
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    return data if isinstance(data, dict) else {"model": data, "feature_cols": None}
            except:
                pass
    return None


def generate_predictions(df: pl.DataFrame, model_info: Dict) -> pl.DataFrame:
    if model_info is None:
        return df.with_columns([pl.lit(0.5).alias("probability")])
    
    model = model_info["model"]
    feature_cols = model_info.get("feature_cols")
    
    if feature_cols is None:
        exclude = ["date", "symbol", "target", "signal", "probability", "predicted_probability"]
        numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in numeric_types]
    
    try:
        X = np.nan_to_num(df.select(feature_cols).to_numpy(), nan=0)
        probs = model.predict_proba(X)[:, 1]
        return df.with_columns([pl.Series("probability", probs)])
    except:
        return df.with_columns([pl.lit(0.5).alias("probability")])


def compute_signals(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["symbol", "date"])
    
    if "sma_50" not in df.columns:
        df = df.with_columns([pl.col("close").rolling_mean(50).over("symbol").alias("sma_50")])
        df = df.with_columns([
            pl.when(pl.col("sma_50").is_null()).then(pl.col("close")).otherwise(pl.col("sma_50")).alias("sma_50")
        ])
    
    return df.with_columns([
        pl.when((pl.col("probability") > 0.70) & (pl.col("close") > pl.col("sma_50"))).then(pl.lit("STRONG_BUY"))
        .when(pl.col("probability") > 0.60).then(pl.lit("BUY"))
        .when(pl.col("probability") < 0.40).then(pl.lit("SELL"))
        .otherwise(pl.lit("HOLD")).alias("signal")
    ])


def prepare_chart_data(df: pl.DataFrame, symbol: str) -> Dict[str, Any]:
    symbol_df = df.filter(pl.col("symbol") == symbol).sort("date")
    if symbol_df.is_empty():
        return {}
    
    dates = symbol_df["date"].cast(pl.Utf8).to_list()
    data = {
        "dates": dates,
        "close": symbol_df["close"].to_list(),
        "open": symbol_df["open"].to_list() if "open" in symbol_df.columns else [],
        "high": symbol_df["high"].to_list() if "high" in symbol_df.columns else [],
        "low": symbol_df["low"].to_list() if "low" in symbol_df.columns else [],
        "volume": symbol_df["volume"].to_list() if "volume" in symbol_df.columns else [],
        "sma_50": symbol_df["sma_50"].to_list() if "sma_50" in symbol_df.columns else [],
        "strong_buy_signals": [], "buy_signals": [], "sell_signals": []
    }
    
    for row in symbol_df.iter_rows(named=True):
        sig = {"date": str(row["date"])[:10], "price": row["close"], "probability": row.get("probability", 0.5)}
        signal = row.get("signal", "HOLD")
        if signal == "STRONG_BUY":
            data["strong_buy_signals"].append(sig)
        elif signal == "BUY":
            data["buy_signals"].append(sig)
        elif signal == "SELL":
            data["sell_signals"].append(sig)
    
    return data


def get_ticker_summary(df: pl.DataFrame, backtest_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
    summaries = []
    
    for symbol in df["symbol"].unique().to_list():
        symbol_df = df.filter(pl.col("symbol") == symbol).sort("date")
        if symbol_df.is_empty():
            continue
        
        latest = symbol_df.tail(1).to_dicts()[0]
        first = symbol_df.head(1).to_dicts()[0]
        change_pct = ((latest["close"] - first["close"]) / first["close"]) * 100
        
        signal_counts = {row["signal"]: row["count"] for row in 
                        symbol_df.group_by("signal").agg(pl.count().alias("count")).iter_rows(named=True)}
        
        summary = {
            "symbol": symbol, "latest_price": round(latest["close"], 2),
            "change_pct": round(change_pct, 2), "latest_signal": latest.get("signal", "HOLD"),
            "probability": round(latest.get("probability", 0.5), 4),
            "total_days": symbol_df.height,
            "strong_buy_count": signal_counts.get("STRONG_BUY", 0),
            "buy_count": signal_counts.get("BUY", 0),
            "sell_count": signal_counts.get("SELL", 0),
            "hold_count": signal_counts.get("HOLD", 0),
            "latest_date": str(latest["date"])[:10],
            "best_model": "N/A", "auc": 0, "win_rate": 0, "total_pnl": 0, "total_trades": 0
        }
        
        if backtest_data:
            if symbol in backtest_data.get("model_metrics", {}):
                m = backtest_data["model_metrics"][symbol]
                summary["best_model"] = m.get("best_model", "N/A")
                summary["auc"] = round(m.get("best_auc", 0), 4)
            if symbol in backtest_data.get("backtest", {}).get("by_ticker", {}):
                bt = backtest_data["backtest"]["by_ticker"][symbol]
                summary["win_rate"] = bt.get("win_rate", 0)
                summary["total_pnl"] = bt.get("total_pnl", 0)
                summary["total_trades"] = bt.get("total_trades", 0)
        
        summaries.append(summary)
    
    summaries.sort(key=lambda x: x["auc"], reverse=True)
    return summaries


def load_backtest_report(base_path: Path) -> Optional[Dict]:
    for path in [base_path / "backtest_report.json", base_path.parent / "backtest_report.json",
                 Path("backtest_report.json"), Path("../backtest_report.json")]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass
    return None


def generate_html_dashboard(df: pl.DataFrame, output_path: str = "dashboard.html", data_path: Path = None) -> str:
    backtest_data = load_backtest_report(data_path) if data_path else None
    symbols = df["symbol"].unique().to_list()
    summaries = get_ticker_summary(df, backtest_data)
    
    all_data = {symbol: prepare_chart_data(df, symbol) for symbol in symbols}
    backtest_global = backtest_data.get("backtest", {}).get("global", {}) if backtest_data else {}
    
    chart_data_json = json.dumps(all_data, default=str)
    summaries_json = json.dumps(summaries)
    symbols_json = json.dumps(symbols)
    backtest_json = json.dumps(backtest_global, default=str)
    
    model_info = {"name": "N/A", "metric": "N/A", "target": "Subida >1% en 5 dias"}
    if backtest_data and "summary" in backtest_data:
        dist = backtest_data["summary"].get("model_distribution", {})
        if dist:
            model_info["name"] = max(dist.items(), key=lambda x: x[1])[0]
            model_info["metric"] = f"AUC: {backtest_data['summary'].get('avg_best_auc', 0):.2%}"
    
    model_info_json = json.dumps(model_info)
    bt_config = backtest_global.get("config", {"tp_pct": 5, "sl_pct": 3, "max_hold_days": 10})
    
    html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Trading - MVP Academico</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --bg-primary: #1a1d21; --bg-secondary: #22262b; --bg-card: #272b30;
            --text-primary: #d1d5db; --text-secondary: #9ca3af; --text-muted: #6b7280;
            --border-color: #374151;
            --color-buy: #22c55e; --color-sell: #ef4444; --color-hold: #eab308;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: var(--bg-primary); color: var(--text-primary); min-height: 100vh; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .header {{ text-align: center; padding: 32px 0; border-bottom: 1px solid var(--border-color); margin-bottom: 32px; }}
        .header h1 {{ font-size: 1.75rem; font-weight: 600; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 16px; }}
        .header .params {{ display: inline-flex; gap: 16px; background: var(--bg-secondary); padding: 8px 16px; 
                          border-radius: 6px; font-size: 0.8rem; color: var(--text-muted); }}
        .strategy-explanation {{ background: var(--bg-secondary); border: 1px solid var(--border-color); 
                                border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        .strategy-explanation h3 {{ font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 16px; }}
        .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .strategy-item {{ padding: 12px; background: var(--bg-primary); border-radius: 6px; }}
        .strategy-signal {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-bottom: 8px; }}
        .strategy-signal.buy {{ background: rgba(34, 197, 94, 0.15); color: var(--color-buy); }}
        .strategy-signal.sell {{ background: rgba(239, 68, 68, 0.15); color: var(--color-sell); }}
        .strategy-signal.hold {{ background: rgba(234, 179, 8, 0.15); color: var(--color-hold); }}
        .strategy-item p {{ font-size: 0.8rem; color: var(--text-muted); }}
        .kpi-section {{ margin-bottom: 32px; }}
        .section-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; 
                         color: var(--text-muted); margin-bottom: 16px; font-weight: 600; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
        .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; }}
        .kpi-card .label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }}
        .kpi-card .value {{ font-size: 1.75rem; font-weight: 600; }}
        .kpi-card .value.positive {{ color: var(--color-buy); }}
        .kpi-card .value.negative {{ color: var(--color-sell); }}
        .model-block {{ background: var(--bg-secondary); border: 1px solid var(--border-color); 
                       border-radius: 8px; padding: 20px; margin-bottom: 32px; }}
        .model-block .model-title {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 12px; }}
        .model-block .model-info {{ display: flex; gap: 24px; flex-wrap: wrap; }}
        .model-block .model-item {{ display: flex; flex-direction: column; }}
        .model-block .model-item .item-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }}
        .model-block .model-item .item-value {{ font-size: 1rem; font-weight: 500; }}
        .model-block .model-note {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color); 
                                   font-size: 0.75rem; color: var(--text-muted); font-style: italic; }}
        .ticker-selector {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; 
                           padding: 16px 20px; margin-bottom: 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
        .ticker-selector label {{ font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }}
        .ticker-dropdown {{ padding: 10px 14px; font-size: 0.9rem; background: var(--bg-primary); color: var(--text-primary); 
                           border: 1px solid var(--border-color); border-radius: 6px; min-width: 180px; }}
        .ticker-info {{ display: flex; gap: 12px; flex-wrap: wrap; margin-left: auto; }}
        .info-badge {{ padding: 6px 12px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }}
        .info-badge.price {{ background: var(--bg-secondary); }}
        .info-badge.signal-BUY, .info-badge.signal-STRONG_BUY {{ background: rgba(34, 197, 94, 0.15); color: var(--color-buy); }}
        .info-badge.signal-SELL {{ background: rgba(239, 68, 68, 0.15); color: var(--color-sell); }}
        .info-badge.signal-HOLD {{ background: rgba(234, 179, 8, 0.15); color: var(--color-hold); }}
        .chart-section {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; 
                         padding: 20px; margin-bottom: 24px; }}
        .chart-section .chart-title {{ font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 16px; }}
        .chart-wrapper {{ position: relative; height: 400px; }}
        .chart-legend {{ display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: var(--text-muted); }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        .legend-dot.price {{ background: #60a5fa; }}
        .legend-dot.sma {{ background: #f59e0b; }}
        .legend-dot.buy {{ background: var(--color-buy); }}
        .legend-dot.sell {{ background: var(--color-sell); }}
        .signals-section {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 12px 8px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; 
             color: var(--text-muted); border-bottom: 1px solid var(--border-color); }}
        td {{ padding: 12px 8px; font-size: 0.85rem; border-bottom: 1px solid var(--border-color); }}
        tr:hover {{ background: var(--bg-secondary); }}
        .signal-badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
        .signal-badge.BUY, .signal-badge.STRONG_BUY {{ background: rgba(34, 197, 94, 0.15); color: var(--color-buy); }}
        .signal-badge.SELL {{ background: rgba(239, 68, 68, 0.15); color: var(--color-sell); }}
        .no-signals {{ text-align: center; padding: 40px; color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Sistema de Trading Algoritmico</h1>
            <p class="subtitle">MVP Academico - Senales con Machine Learning y Backtesting</p>
            <div class="params">
                <span>TP: {bt_config.get('tp_pct', 5)}%</span>
                <span>SL: {bt_config.get('sl_pct', 3)}%</span>
                <span>Horizonte: {bt_config.get('max_hold_days', 10)} dias</span>
            </div>
        </header>
        
        <div class="strategy-explanation">
            <h3>Como funciona la estrategia</h3>
            <div class="strategy-grid">
                <div class="strategy-item">
                    <span class="strategy-signal buy">BUY</span>
                    <p>Probabilidad de subida &gt;60%. Predice subida &gt;1% en 5 dias.</p>
                </div>
                <div class="strategy-item">
                    <span class="strategy-signal sell">SELL</span>
                    <p>Probabilidad de subida &lt;40%. Predice caida o estancamiento.</p>
                </div>
                <div class="strategy-item">
                    <span class="strategy-signal hold">HOLD</span>
                    <p>Probabilidad 40-60%. Incertidumbre alta.</p>
                </div>
            </div>
        </div>
        
        <section class="kpi-section">
            <h2 class="section-title">Metricas de Estrategia</h2>
            <div class="kpi-grid" id="kpiGrid"></div>
        </section>
        
        <div class="model-block" id="modelBlock"></div>
        
        <div class="ticker-selector">
            <label for="tickerSelect">Ticker:</label>
            <select id="tickerSelect" class="ticker-dropdown"></select>
            <div class="ticker-info" id="tickerInfo"></div>
        </div>
        
        <section class="chart-section">
            <h3 class="chart-title">Precio Historico y Senales</h3>
            <div class="chart-wrapper"><canvas id="priceChart"></canvas></div>
            <div class="chart-legend">
                <div class="legend-item"><div class="legend-dot price"></div><span>Precio</span></div>
                <div class="legend-item"><div class="legend-dot sma"></div><span>SMA 50</span></div>
                <div class="legend-item"><div class="legend-dot buy"></div><span>BUY</span></div>
                <div class="legend-item"><div class="legend-dot sell"></div><span>SELL</span></div>
            </div>
        </section>
        
        <section class="signals-section">
            <h3 class="section-title" style="margin-bottom:16px">Historial de Senales</h3>
            <div id="signalsTable"><p class="no-signals">Selecciona un ticker</p></div>
        </section>
    </div>

    <script>
        const allData = {chart_data_json};
        const summaries = {summaries_json};
        const symbols = {symbols_json};
        const backtestGlobal = {backtest_json};
        const modelInfo = {model_info_json};
        let priceChart = null;
        
        function init() {{ renderKPIs(); renderModelBlock(); initDropdown(); }}
        
        function renderKPIs() {{
            const bt = backtestGlobal, container = document.getElementById('kpiGrid');
            if (!bt || !bt.total_trades) {{ container.innerHTML = '<p style="color:var(--text-muted)">Sin datos de backtest</p>'; return; }}
            const kpis = [
                {{ label: 'Return Total', value: `${{bt.total_pnl >= 0 ? '+' : ''}}${{bt.total_pnl.toFixed(1)}}%`, colorClass: bt.total_pnl >= 0 ? 'positive' : 'negative' }},
                {{ label: 'Win Rate', value: `${{bt.win_rate}}%`, colorClass: bt.win_rate >= 50 ? 'positive' : 'negative' }},
                {{ label: 'Trades', value: bt.total_trades, colorClass: '' }},
                {{ label: 'Max Drawdown', value: `-${{(bt.max_drawdown||0).toFixed(1)}}%`, colorClass: 'negative' }},
                {{ label: 'Sharpe', value: (bt.sharpe_ratio||0).toFixed(2), colorClass: bt.sharpe_ratio > 1 ? 'positive' : '' }}
            ];
            container.innerHTML = kpis.map(k => `<div class="kpi-card"><div class="label">${{k.label}}</div><div class="value ${{k.colorClass}}">${{k.value}}</div></div>`).join('');
        }}
        
        function renderModelBlock() {{
            const container = document.getElementById('modelBlock');
            let bestModel = 'N/A', avgAuc = 0;
            if (summaries.length) {{
                const counts = {{}};
                summaries.forEach(s => {{ if (s.best_model && s.best_model !== 'N/A') counts[s.best_model] = (counts[s.best_model]||0)+1; }});
                bestModel = Object.keys(counts).sort((a,b) => counts[b]-counts[a])[0] || 'N/A';
                const aucs = summaries.filter(s => s.auc > 0).map(s => s.auc);
                avgAuc = aucs.length ? aucs.reduce((a,b) => a+b, 0) / aucs.length : 0;
            }}
            container.innerHTML = `
                <span class="model-title">Modelo ML</span>
                <div class="model-info">
                    <div class="model-item"><span class="item-label">Algoritmo</span><span class="item-value">${{bestModel}}</span></div>
                    <div class="model-item"><span class="item-label">AUC Promedio</span><span class="item-value">${{(avgAuc*100).toFixed(1)}}%</span></div>
                    <div class="model-item"><span class="item-label">Target</span><span class="item-value">Subida >1% en 5 dias</span></div>
                </div>
                <p class="model-note">Predice probabilidad de subida. Senales combinan probabilidad + indicadores tecnicos (SMA).</p>`;
        }}
        
        function initDropdown() {{
            const select = document.getElementById('tickerSelect');
            select.innerHTML = summaries.map(s => `<option value="${{s.symbol}}">${{s.symbol}} (${{s.latest_signal}})</option>`).join('');
            select.addEventListener('change', e => updateDashboard(e.target.value));
            if (summaries.length) updateDashboard(summaries[0].symbol);
        }}
        
        function updateDashboard(symbol) {{
            const data = allData[symbol], summary = summaries.find(s => s.symbol === symbol);
            if (!data || !summary) return;
            document.getElementById('tickerInfo').innerHTML = `
                <span class="info-badge price">$${{summary.latest_price.toFixed(2)}}</span>
                <span class="info-badge signal-${{summary.latest_signal}}">${{summary.latest_signal}}</span>
                <span class="info-badge price">Prob: ${{(summary.probability*100).toFixed(0)}}%</span>`;
            updateChart(data);
            updateSignalsTable(data);
        }}
        
        function updateChart(data) {{
            if (priceChart) priceChart.destroy();
            const labels = data.dates.map(d => d.substring(0, 10));
            const buyPts = [...data.strong_buy_signals, ...data.buy_signals].map(s => ({{ x: s.date, y: s.price }}));
            const sellPts = data.sell_signals.map(s => ({{ x: s.date, y: s.price }}));
            priceChart = new Chart(document.getElementById('priceChart').getContext('2d'), {{
                type: 'line',
                data: {{
                    labels,
                    datasets: [
                        {{ label: 'Precio', data: data.close, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.05)', fill: true, tension: 0.1, pointRadius: 0, borderWidth: 1.5 }},
                        {{ label: 'SMA 50', data: data.sma_50, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, borderDash: [4,4], fill: false }},
                        {{ label: 'BUY', data: buyPts, backgroundColor: '#22c55e', borderColor: '#22c55e', pointRadius: 6, pointStyle: 'triangle', showLine: false }},
                        {{ label: 'SELL', data: sellPts, backgroundColor: '#ef4444', borderColor: '#ef4444', pointRadius: 6, pointStyle: 'triangle', rotation: 180, showLine: false }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(55,61,71,0.5)' }}, ticks: {{ color: '#6e7681', maxTicksLimit: 10 }} }},
                        y: {{ grid: {{ color: 'rgba(55,61,71,0.5)' }}, ticks: {{ color: '#6e7681', callback: v => '$' + v.toFixed(2) }} }}
                    }}
                }}
            }});
        }}
        
        function updateSignalsTable(data) {{
            const container = document.getElementById('signalsTable');
            const signals = [...data.strong_buy_signals.map(s => ({{...s, signal:'STRONG_BUY'}})), 
                            ...data.buy_signals.map(s => ({{...s, signal:'BUY'}})),
                            ...data.sell_signals.map(s => ({{...s, signal:'SELL'}}))].sort((a,b) => new Date(b.date)-new Date(a.date));
            if (!signals.length) {{ container.innerHTML = '<p class="no-signals">Sin senales activas</p>'; return; }}
            container.innerHTML = `<table><thead><tr><th>Fecha</th><th>Senal</th><th>Precio</th><th>Prob</th></tr></thead><tbody>` +
                signals.slice(0,20).map(s => `<tr><td>${{s.date}}</td><td><span class="signal-badge ${{s.signal}}">${{s.signal.replace('_',' ')}}</span></td><td>$${{s.price.toFixed(2)}}</td><td>${{(s.probability*100).toFixed(0)}}%</td></tr>`).join('') +
                `</tbody></table>` + (signals.length > 20 ? `<p style="text-align:center;color:var(--text-muted);margin-top:12px;font-size:0.8rem">Mostrando 20 de ${{signals.length}}</p>` : '');
        }}
        
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"dashboard: {output_path}")
    return output_path


def create_dashboard(data_path: str = "data", output_path: str = "dashboard.html") -> str:
    data_path = Path(data_path)
    
    logger.info("cargando datos...")
    df = load_training_data(data_path)
    
    logger.info("cargando modelo...")
    model_info = load_model(data_path)
    
    logger.info("generando predicciones...")
    df = generate_predictions(df, model_info)
    
    logger.info("calculando señales...")
    df = compute_signals(df)
    
    return generate_html_dashboard(df, output_path, data_path)


if __name__ == "__main__":
    import sys
    
    if Path("data").exists():
        data_path = "data"
    elif Path("../data").exists():
        data_path = "../data"
    else:
        print("error: no se encuentra 'data'")
        sys.exit(1)
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dashboard.html"
    print(f"dashboard: {create_dashboard(data_path, output_path)}")

"""
screener de stocks con yahoo finance api
metodos: api json (principal), fallback con datos ejemplo
"""

import requests
import polars as pl
from typing import List, Dict, Optional, Any, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScreenerError(Exception):
    pass


SCREENER_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

AVAILABLE_SCREENERS = {
    "small_cap_gainers": "Small Cap Gainers",
    "day_gainers": "Day Gainers", 
    "day_losers": "Day Losers",
    "most_actives": "Most Actives",
    "undervalued_large_caps": "Undervalued Large Caps",
    "growth_technology_stocks": "Growth Technology Stocks",
}


def fetch_screener_api(screener_id: str = "small_cap_gainers", count: int = 25, timeout: int = 15) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """obtiene datos del screener via api json de yahoo finance"""
    params = {"scrIds": screener_id, "count": count}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "application/json"}
    
    try:
        logger.info(f"consultando api: {screener_id}")
        response = requests.get(SCREENER_ENDPOINT, params=params, headers=headers, timeout=timeout)
        
        if response.status_code == 404:
            return [], f"screener '{screener_id}' no encontrado (404)"
        elif response.status_code == 403:
            return [], "acceso denegado (403)"
        
        response.raise_for_status()
        data = response.json()
        
        if "finance" not in data:
            return [], "respuesta sin campo 'finance'"
        if "result" not in data["finance"] or not data["finance"]["result"]:
            return [], "respuesta sin resultados"
        
        result = data["finance"]["result"][0]
        logger.info(f"screener: {result.get('title', 'N/A')}, total: {result.get('total', 0)}")
        
        quotes = result.get("quotes", [])
        if not quotes:
            return [], "no se encontraron quotes"
        
        logger.info(f"quotes obtenidos: {len(quotes)}")
        return quotes, None
        
    except requests.Timeout:
        return [], f"timeout {timeout}s"
    except requests.ConnectionError as e:
        return [], f"error conexion: {e}"
    except requests.RequestException as e:
        return [], f"error request: {e}"
    except Exception as e:
        return [], f"error: {e}"


def parse_quotes_to_stocks(quotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """convierte quotes de la api a formato estandarizado"""
    stocks = []
    for quote in quotes:
        try:
            symbol = quote.get("symbol", "")
            if not symbol:
                continue
            
            stocks.append({
                "symbol": symbol,
                "name": quote.get("shortName", quote.get("longName", symbol)),
                "price": float(quote.get("regularMarketPrice", 0) or 0),
                "change_pct": float(quote.get("regularMarketChangePercent", 0) or 0),
                "volume": int(quote.get("regularMarketVolume", 0) or 0),
                "market_cap": int(quote.get("marketCap", 0) or 0)
            })
        except (ValueError, TypeError):
            continue
    return stocks


def get_fallback_stocks() -> List[Dict[str, Any]]:
    """datos de fallback predefinidos"""
    return [
        {"symbol": "SOFI", "name": "SoFi Technologies", "price": 8.50, "change_pct": 2.3, "volume": 45000000, "market_cap": 8000000000},
        {"symbol": "PLTR", "name": "Palantir Technologies", "price": 22.00, "change_pct": 1.5, "volume": 35000000, "market_cap": 48000000000},
        {"symbol": "NIO", "name": "NIO Inc", "price": 5.20, "change_pct": -1.2, "volume": 40000000, "market_cap": 10000000000},
        {"symbol": "RIVN", "name": "Rivian Automotive", "price": 12.00, "change_pct": 3.1, "volume": 25000000, "market_cap": 12000000000},
        {"symbol": "LCID", "name": "Lucid Group", "price": 3.50, "change_pct": -0.8, "volume": 30000000, "market_cap": 8000000000},
        {"symbol": "HOOD", "name": "Robinhood Markets", "price": 15.00, "change_pct": 4.2, "volume": 20000000, "market_cap": 13000000000},
        {"symbol": "SNAP", "name": "Snap Inc", "price": 11.00, "change_pct": 0.5, "volume": 18000000, "market_cap": 18000000000},
        {"symbol": "PINS", "name": "Pinterest", "price": 32.00, "change_pct": 1.8, "volume": 12000000, "market_cap": 22000000000},
        {"symbol": "DKNG", "name": "DraftKings", "price": 38.00, "change_pct": 2.0, "volume": 8000000, "market_cap": 18000000000},
        {"symbol": "AMD", "name": "Advanced Micro Devices", "price": 120.00, "change_pct": 1.2, "volume": 50000000, "market_cap": 195000000000},
    ]


def get_small_cap_screener(screener_id: str = "small_cap_gainers", count: int = 25, 
                           min_price: float = 0.0, max_price: float = 10000.0, 
                           use_fallback: bool = True) -> pl.DataFrame:
    """obtiene screener con fallbacks"""
    stocks = []
    method_used = "ninguno"
    
    screeners_to_try = [screener_id]
    if screener_id == "small_cap_gainers":
        screeners_to_try.extend(["day_gainers", "most_actives"])
    
    logger.info("=" * 50)
    logger.info("METODO 1: API Yahoo Finance")
    
    for scr_id in screeners_to_try:
        quotes, error = fetch_screener_api(scr_id, count=count)
        if error:
            logger.warning(f"error con {scr_id}: {error}")
            continue
        if quotes:
            stocks = parse_quotes_to_stocks(quotes)
            if stocks:
                method_used = f"api ({scr_id})"
                logger.info(f"exito: {len(stocks)} stocks")
                break
    
    if not stocks:
        logger.info("=" * 50)
        logger.info("METODO 2: Fallback")
        if not use_fallback:
            raise ScreenerError("no se pudieron obtener datos y fallback deshabilitado")
        stocks = get_fallback_stocks()
        method_used = "fallback"
        logger.warning(f"usando fallback: {len(stocks)} stocks")
    
    df = pl.DataFrame(stocks)
    logger.info(f"datos via {method_used}: {len(df)} stocks")
    
    if min_price > 0 or max_price < 10000:
        df_filtered = df.filter((pl.col("price") >= min_price) & (pl.col("price") <= max_price))
        if not df_filtered.is_empty():
            df = df_filtered
    
    logger.info(f"screener completado: {len(df)} stocks")
    return df


def save_screener_results(df: pl.DataFrame, output_path: str) -> None:
    if df.is_empty():
        raise ScreenerError("no se puede guardar screener vacio")
    df.write_csv(output_path)
    logger.info(f"guardado: {output_path}")


def test_screener() -> None:
    print("\n" + "=" * 60)
    print("TEST SCREENER")
    print("=" * 60)
    try:
        df = get_small_cap_screener(count=10)
        print(f"ok: {len(df)} stocks")
        for row in df.iter_rows(named=True):
            print(f"  {row['symbol']:6s} | {row['name'][:25]:25s} | ${row['price']:8.2f}")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    test_screener()

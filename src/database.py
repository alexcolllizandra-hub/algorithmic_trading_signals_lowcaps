"""
base de datos mongodb para el sistema de trading
persiste: datos de mercado, señales, backtest, metricas
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "trading_mvp"

COLLECTIONS = {
    "market_data": "market_data",
    "signals": "signals",
    "backtest": "backtest_results",
    "models": "model_metrics",
    "config": "system_config"
}


class MongoDBConnection:
    """conexion singleton a mongodb"""
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self, uri: str = MONGO_URI, db_name: str = DB_NAME) -> bool:
        try:
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self._client.admin.command('ping')
            self._db = self._client[db_name]
            logger.info(f"conectado a mongodb: {db_name}")
            self._create_indexes()
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"mongodb no disponible: {e}")
            return False
    
    def _create_indexes(self) -> None:
        if self._db is None:
            return
        self._db[COLLECTIONS["market_data"]].create_index([("symbol", ASCENDING), ("date", DESCENDING)], unique=True)
        self._db[COLLECTIONS["signals"]].create_index([("symbol", ASCENDING), ("date", DESCENDING)])
        self._db[COLLECTIONS["signals"]].create_index([("signal", ASCENDING)])
        self._db[COLLECTIONS["backtest"]].create_index([("run_date", DESCENDING)])
    
    @property
    def db(self):
        return self._db
    
    @property
    def is_connected(self) -> bool:
        return self._db is not None
    
    def close(self) -> None:
        if self._client:
            self._client.close()
            self._db = None


mongo_conn = MongoDBConnection()


def save_market_data(df: pl.DataFrame) -> int:
    if not mongo_conn.is_connected:
        return 0
    
    collection = mongo_conn.db[COLLECTIONS["market_data"]]
    count = 0
    
    for row in df.iter_rows(named=True):
        doc = {
            "symbol": row["symbol"],
            "date": row["date"] if isinstance(row["date"], datetime) else datetime.fromisoformat(str(row["date"])[:10]),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": int(row.get("volume", 0) or 0),
            "updated_at": datetime.now()
        }
        collection.update_one({"symbol": doc["symbol"], "date": doc["date"]}, {"$set": doc}, upsert=True)
        count += 1
    
    logger.info(f"market data: {count} registros")
    return count


def load_market_data(symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> Optional[pl.DataFrame]:
    if not mongo_conn.is_connected:
        return None
    
    collection = mongo_conn.db[COLLECTIONS["market_data"]]
    query = {}
    
    if symbols:
        query["symbol"] = {"$in": symbols}
    if start_date or end_date:
        query["date"] = {}
        if start_date:
            query["date"]["$gte"] = start_date
        if end_date:
            query["date"]["$lte"] = end_date
    
    docs = list(collection.find(query).sort([("symbol", 1), ("date", 1)]))
    if not docs:
        return None
    
    return pl.DataFrame([{
        "date": doc["date"], "symbol": doc["symbol"],
        "open": doc["open"], "high": doc["high"], "low": doc["low"],
        "close": doc["close"], "volume": doc.get("volume", 0)
    } for doc in docs])


def save_signals(df: pl.DataFrame) -> int:
    if not mongo_conn.is_connected:
        return 0
    
    collection = mongo_conn.db[COLLECTIONS["signals"]]
    count = 0
    
    for row in df.iter_rows(named=True):
        doc = {
            "symbol": row["symbol"],
            "date": row["date"] if isinstance(row["date"], datetime) else datetime.fromisoformat(str(row["date"])[:10]),
            "signal": row.get("signal", "HOLD"),
            "probability": float(row.get("probability", 0.5)),
            "price": float(row["close"]),
            "created_at": datetime.now()
        }
        collection.update_one({"symbol": doc["symbol"], "date": doc["date"]}, {"$set": doc}, upsert=True)
        count += 1
    
    logger.info(f"señales: {count}")
    return count


def get_latest_signals(limit: int = 50) -> List[Dict]:
    if not mongo_conn.is_connected:
        return []
    return list(mongo_conn.db[COLLECTIONS["signals"]].find({"signal": {"$ne": "HOLD"}}).sort("date", DESCENDING).limit(limit))


def save_backtest_results(results: Dict[str, Any]) -> bool:
    if not mongo_conn.is_connected:
        return False
    
    mongo_conn.db[COLLECTIONS["backtest"]].insert_one({
        "run_date": datetime.now(),
        "global_stats": results.get("global", {}),
        "by_ticker": results.get("by_ticker", {}),
        "config": results.get("config", {})
    })
    logger.info("backtest guardado")
    return True


def get_latest_backtest() -> Optional[Dict]:
    if not mongo_conn.is_connected:
        return None
    return mongo_conn.db[COLLECTIONS["backtest"]].find_one(sort=[("run_date", DESCENDING)])


def save_model_metrics(metrics: Dict[str, Any]) -> bool:
    if not mongo_conn.is_connected:
        return False
    mongo_conn.db[COLLECTIONS["models"]].insert_one({"run_date": datetime.now(), "metrics": metrics})
    return True


def get_database_stats() -> Dict[str, Any]:
    if not mongo_conn.is_connected:
        return {"connected": False}
    return {
        "connected": True,
        "collections": {name: mongo_conn.db[coll].count_documents({}) for name, coll in COLLECTIONS.items()}
    }


def clear_collection(collection_name: str) -> int:
    if not mongo_conn.is_connected or collection_name not in COLLECTIONS.values():
        return 0
    result = mongo_conn.db[collection_name].delete_many({})
    return result.deleted_count


def init_mongodb(uri: str = MONGO_URI, db_name: str = DB_NAME) -> bool:
    return mongo_conn.connect(uri, db_name)


if __name__ == "__main__":
    if init_mongodb():
        print(f"stats: {get_database_stats()}")
    else:
        print("no se pudo conectar")

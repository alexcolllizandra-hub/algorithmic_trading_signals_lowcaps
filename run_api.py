"""
lanza la api de trading en local
ejecutar: python run_api.py
api disponible en: http://localhost:8000
documentacion: http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

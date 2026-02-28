from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np
from typing import List, Optional

# Internal module
from risk_engine import preprocess_portfolio_data, get_risk_metrics

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="Stateless Backend for Dynamic Quantitative Risk Analytics",
    version="1.3.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """ Verify backend state """
    return {
        "status": "online",
        "engine": "dynamic_benchmark_risk_analytics",
        "version": "1.3.0"
    }

@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    benchmark_ticker: str = Form(...), # Required field
    weights: str = Form(None),         # Optional JSON array: [0.3, 0.7]
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000)
):
    """
    Analyzes portfolio risk by fetching real-time benchmark data.
    Input: CSV (Date, Asset1, Asset2...) + benchmark_ticker.
    """
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Structure Validation
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data. CSV requires at least 30 rows.")
            
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Invalid structure. CSV must have at least one Date and one Asset column.")
            
        # 3. Preprocessing (Assets-only + yfinance benchmark)
        asset_returns, bench_returns, asset_names = preprocess_portfolio_data(df, benchmark_ticker)
        
        # 4. Handle Portfolio Weights
        processed_weights = None
        if weights:
            try:
                processed_weights = json.loads(weights)
                if not isinstance(processed_weights, list):
                    raise ValueError
                if len(processed_weights) != len(asset_names):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Weight count ({len(processed_weights)}) does not match Asset count ({len(asset_names)})"
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Weights must be a valid JSON array of numbers.")

        # 5. Core Computation
        results = get_risk_metrics(
            asset_returns=asset_returns, 
            bench_returns=bench_returns, 
            benchmark_ticker=benchmark_ticker,
            asset_names=asset_names,
            weights=processed_weights,
            risk_free_rate=risk_free_rate, 
            confidence_level=confidence_level,
            simulations=simulations
        )
        
        return results
        
    except HTTPException as he:
        # Re-raise controlled HTTP exceptions
        raise he
    except Exception as e:
        # Catch unexpected computational or fetching errors
        return JSONResponse(
            status_code=400,
            content={"error": "Analytical failure", "detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    # Bound to port 8000 for local dev and cloud deployment visibility
    uvicorn.run(app, host="0.0.0.0", port=8000)
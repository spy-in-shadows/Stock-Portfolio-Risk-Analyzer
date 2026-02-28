from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np
from typing import List, Optional

# Internal module
from risk_engine import preprocess_blended_data, get_risk_metrics

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="Blended Benchmark Quantitative Risk Engine",
    version="1.2.0"
)

# Enable CORS for frontend (Vercel)
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
        "engine": "blended_benchmark_risk_analytics",
        "version": "1.2.0"
    }

@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    benchmark_tickers: str = Form(None), # JSON array: ["^NSEI", "^NSEMDCP50"]
    benchmark_weights: str = Form(None), # JSON array: [0.7, 0.3]
    weights: str = Form(None),           # JSON array: [0.33, 0.33, 0.34]
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000)
):
    """
    Primary endpoint for risk calculation with Blended Benchmark support.
    Logic: Uses 'Benchmark' column in CSV first, else builds blended from yfinance.
    """
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Basic Validation
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data. CSV requires at least 30 rows.")
            
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Invalid structure. CSV must have at least a Date and one Asset column.")
            
        # 3. Handle Inputs Parsing
        tickers = None
        bench_weights = None
        if benchmark_tickers:
            try:
                tickers = json.loads(benchmark_tickers)
                if not isinstance(tickers, list) or len(tickers) == 0:
                    raise ValueError
            except ValueError:
                raise HTTPException(status_code=400, detail="benchmark_tickers must be a non-empty JSON array of strings.")
        
        if benchmark_weights:
            try:
                bench_weights = json.loads(benchmark_weights)
                if not isinstance(bench_weights, list):
                    raise ValueError
            except ValueError:
                raise HTTPException(status_code=400, detail="benchmark_weights must be a JSON array of numbers.")
        
        # Check lengths if both provided
        if tickers and bench_weights and len(tickers) != len(bench_weights):
            raise HTTPException(status_code=400, detail="benchmark_tickers and benchmark_weights must have the same length.")

        # 4. Preprocessing (Blended Logic)
        asset_returns, bench_returns, source, b_type, b_components, asset_names = preprocess_blended_data(
            df, benchmark_tickers=tickers, benchmark_weights=bench_weights
        )
        
        # 5. Handle Portfolio Weights
        processed_port_weights = None
        if weights:
            try:
                processed_port_weights = json.loads(weights)
                if not isinstance(processed_port_weights, list):
                    raise ValueError
                if len(processed_port_weights) != len(asset_names):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Portfolio weight count mismatch with asset count ({len(asset_names)})"
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Portfolio weights must be a valid JSON array of numbers.")

        # 6. Core Computation
        results = get_risk_metrics(
            asset_returns, 
            bench_returns, 
            benchmark_source=source,
            benchmark_type=b_type,
            benchmark_components=b_components,
            asset_names=asset_names,
            weights=processed_port_weights,
            risk_free_rate=risk_free_rate, 
            confidence_level=confidence_level,
            simulations=simulations
        )
        
        return results
        
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Analytical Failure", "detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
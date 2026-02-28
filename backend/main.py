from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np

# Internal module
from risk_engine import preprocess_hybrid_data, get_risk_metrics

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="Hybrid Benchmark Quantitative Risk Engine",
    version="1.1.0"
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
        "engine": "hybrid_benchmark_risk_analytics",
        "version": "1.1.0"
    }

@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    benchmark_ticker: str = Form(None), # Optional yfinance ticker
    weights: str = Form(None),         # JSON array: [0.33, 0.33, 0.34]
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000)
):
    """
    Primary endpoint for risk calculation with Hybrid Benchmark support.
    Logic: Uses 'Benchmark' column from CSV if present, else fetches from yfinance.
    """
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Basic Validation (min 30 rows of data + Date column)
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data. CSV requires at least 30 rows.")
            
        if len(df.columns) < 2: # Date + 1 Asset
            raise HTTPException(status_code=400, detail="Invalid structure. CSV must have at least a Date and one Asset column.")
            
        # 3. Preprocessing (Hybrid Benchmark Logic)
        asset_returns, bench_returns, source, asset_names = preprocess_hybrid_data(df, benchmark_ticker)
        
        # 4. Handle Weights
        processed_weights = None
        if weights:
            try:
                processed_weights = json.loads(weights)
                if not isinstance(processed_weights, list):
                    raise ValueError
                if len(processed_weights) != len(asset_names):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Weight count ({len(processed_weights)}) mismatch with Asset count ({len(asset_names)})"
                    )
                # Check for sum proximitiy to 1.0 (vectorized check handled in risk_engine)
            except ValueError:
                raise HTTPException(status_code=400, detail="Weights must be a valid JSON array of numbers.")

        # 5. Core Computation
        results = get_risk_metrics(
            asset_returns, 
            bench_returns, 
            benchmark_source=source,
            asset_names=asset_names,
            weights=processed_weights,
            risk_free_rate=risk_free_rate, 
            confidence_level=confidence_level,
            simulations=simulations
        )
        
        return results
        
    except HTTPException as he:
        # Re-raise standard FastAPI HTTP exceptions
        raise he
    except Exception as e:
        # General catch-all for calculation or parsing errors
        return JSONResponse(
            status_code=400,
            content={"error": "Analytical Failure", "detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    # Cloud deployment binding
    uvicorn.run(app, host="0.0.0.0", port=10000)
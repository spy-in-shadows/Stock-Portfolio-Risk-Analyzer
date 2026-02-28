from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np

# Internal module
from risk_engine import preprocess_data, get_risk_metrics

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="Stateless Backend for Quantitative Risk Analytics",
    version="1.0.0"
)

# Enable CORS for frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, set to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """ Verify backend state """
    return {
        "status": "online",
        "engine": "stateless_risk_analytics",
        "version": "1.0.0"
    }

@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    weights: str = Form(None), # JSON string representation: [0.33, 0.33, 0.34]
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000)
):
    """
    Primary endpoint for risk calculation.
    Expects CSV with Date (1st), Assets, and Benchmark (last).
    """
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Basic Validation
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data. CSV requires at least 30 rows.")
            
        if len(df.columns) < 3: # Date + 1 Asset + Benchmark
            raise HTTPException(status_code=400, detail="Invalid structure. CSV must have Date, at least one Asset, and a Benchmark column.")
            
        # 3. Preprocessing
        assets_df, benchmark_series = preprocess_data(df)
        
        # 4. Handle Weights
        processed_weights = None
        if weights:
            try:
                processed_weights = json.loads(weights)
                if not isinstance(processed_weights, list):
                    raise ValueError
                if len(processed_weights) != assets_df.shape[1]:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Weight count ({len(processed_weights)}) mismatch with Asset count ({assets_df.shape[1]})"
                    )
                if not np.isclose(np.sum(processed_weights), 1.0, atol=1e-3):
                    raise HTTPException(status_code=400, detail="Weights must sum to approximately 1.0")
            except ValueError:
                raise HTTPException(status_code=400, detail="Weights must be a valid JSON array of numbers.")

        # 5. Core Computation
        results = get_risk_metrics(
            assets_df, 
            benchmark_series, 
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
            content={"error": "Computation failure", "detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
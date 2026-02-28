from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from risk_engine import preprocess_broker_data, get_risk_metrics

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description=(
        "Accepts broker-style holdings CSV exports. "
        "Fetches historical prices via yfinance. "
        "Computes institutional-grade quantitative risk analytics."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "broker_csv_risk_analytics",
        "version": "2.0.0"
    }


@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    benchmark_ticker: str = Form(...),        # Required: e.g. "^NSEI"
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000),
    history_days: int = Form(365)             # How far back to fetch (default: 1 year)
):
    """
    Accepts a broker-style holdings CSV (Ticker, Portfolio Weight %, ...).
    Automatically fetches all historical prices from yfinance.
    Returns full quantitative risk report.
    """
    try:
        # 1. Read uploaded CSV
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as parse_err:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(parse_err)}")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        if len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV has fewer than 2 columns. Ensure it is a valid broker export."
            )

        # 2. Parse holdings + fetch prices + align benchmark
        asset_returns, bench_returns, weights, asset_names, portfolio_value = preprocess_broker_data(
            df=df,
            benchmark_ticker=benchmark_ticker,
            history_days=history_days
        )

        # 3. Run risk engine
        results = get_risk_metrics(
            asset_returns=asset_returns,
            bench_returns=bench_returns,
            benchmark_ticker=benchmark_ticker,
            asset_names=asset_names,
            weights=weights,
            portfolio_value=portfolio_value,
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
            content={"error": "Analytical failure", "detail": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
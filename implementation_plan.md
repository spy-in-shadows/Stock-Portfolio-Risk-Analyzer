# Portfolio Risk Analyzer Implementation Plan

## 1. Roadmap & Implementation Checklist

### Phase 1: Stateless Risk Engine Implementation (`risk_engine.py`) [COMPLETED]
- [x] **Step 1: Data Preprocessing Utility**
- [x] **Step 2: core Statistical Functions**
- [x] **Step 3: Value at Risk (VaR) Suite**
- [x] **Step 4: Monte Carlo Simulation Module**
- [x] **Step 5: Systematic Risk Engine**

### Phase 2: FastAPI Backend Development (`main.py`) [COMPLETED]
- [x] **Endpoint: `GET /`**
- [x] **Endpoint: `POST /analyze`**
- [x] **Validation Layer**
- [x] **JSON Response Formatting**

### Phase 3: Infrastructure & Deployment
- [ ] **Dependencies**: Finalize `requirements.txt`. [DONE]
- [ ] **Runtime Configuration**: Configure `.env` for API base URL.
- [ ] **Deploy Backend**: Setup on Render.
- [ ] **Deploy Frontend**: Setup on Vercel.

### Phase 4: Frontend-Backend Integration [COMPLETED]
- [x] **API Service Layer**: Implement axios/fetch calls to the backend.
- [x] **File Upload Hook**: Connect `UploadPortfolio` to the `/analyze` endpoint.
- [x] **Dynamic Visualization**:
  - Feed Recharts with Monte Carlo simulation paths.
  - Populate Heatmap with Correlation Matrix.
  - Dynamic Gauge for VaR results.
- [x] **Error Handling**: Display backend validation errors on UI toast/banners.

## 2. Technical Stack
- **Framework**: FastAPI
- **Computations**: NumPy, Pandas, SciPy
- **Simulation**: Multivariate Normal Sampling + Cholesky Decomposition
- **Performance**: Vectorized matrix operations for <200ms latency.

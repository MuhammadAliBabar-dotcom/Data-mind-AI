# DataMind AI — Hybrid Data Analytics & Machine Learning

DataMind AI keeps the original single-page HTML/CSS/JavaScript workspace and adds an optional Flask backend for real server-side SQL, machine learning and Gemini-assisted explanations. Browser-side profiling, visualizations, local SQL fallback, cleaning and report export remain available.

**Project version:** 2.0.0

## Main features

1. CSV / Excel upload, validation and preview (10 MB default). The backend allows a larger JSON request limit because parsed JSON can be bigger than the source file.
2. Dataset profiling: data types, missing values, duplicates, unique values and descriptive statistics.
3. Natural-language data chat with a deterministic SQL/result fallback; optional Gemini wording based only on computed results.
4. Read-only SQL execution in an isolated in-memory SQLite database, protected by a SQLite authorizer.
5. Automatic visualizations and computed insights.
6. Cleaning operations that preserve the original dataset.
7. PDF and Excel exports.
8. **Machine Learning Lab:** Random Forest Regression, Isolation Forest anomaly detection and K-Means clustering.

## Local setup (Windows / VS Code)

### 1. Install frontend dependencies

Install Node.js LTS, open the project folder in VS Code Terminal, then run:

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend automatically looks for the Flask API at `http://127.0.0.1:5000`.

### 2. Install and start the backend

Use Python 3.11 or newer (Python 3.13 is supported by the pinned ML stack). In a second terminal, from the project root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
copy backend/.env.example backend/.env
python backend/app.py
```

The backend listens on `http://127.0.0.1:5000`. `GET /api/health` reports backend, scikit-learn and Gemini-key availability.

### 3. Gemini (optional)

Add your Google AI Studio API key to `backend/.env` as `GEMINI_API_KEY=...`, restart Flask, and keep that key on the backend only. DataMind uses Google's Gemini REST endpoint with `gemini-2.5-flash` by default. If the key is missing or the provider is unavailable, DataMind falls back to deterministic local calculations. Gemini may receive the user's question and a small computed result sample; do not enable it for confidential data unless your data policy permits that transfer.

## Connect a separately hosted backend

Open **Settings → Backend Connection** in DataMind AI and enter the backend base URL, for example `https://your-api-host.example.com`. The backend's `CORS_ORIGINS` must contain the exact frontend origin, e.g. `https://your-frontend.example.com` (no wildcard in production). The frontend sends the current dataset to that server when a dataset is loaded. Use HTTPS for both frontend and backend.

### Render backend deployment

The included `render.yaml` installs Gunicorn separately for the Linux host; Gunicorn is intentionally not in the local Windows requirements. It is a starting point. Set `CORS_ORIGINS` to your deployed frontend origin and optionally set `GEMINI_API_KEY` in the Render environment settings. The start command intentionally uses one Gunicorn worker because uploaded data and sessions are currently held in process memory.

The Vercel configuration builds the static frontend only. Deploy the Flask API separately, then enter its URL in Settings.

## Machine-learning model details

- **Prediction:** RandomForestRegressor with median/mode imputation and one-hot encoding. It uses an 80/20 train/test split and reports MAE, RMSE and R². Predictions shown are hold-out estimates, not a time-series forecast.
- **Anomaly detection:** IsolationForest identifies records that differ from the overall numeric pattern. Flags are review signals, not proof of fraud or incorrect data.
- **Segmentation:** KMeans clusters rows using selected numeric features after imputation and standardization. Without a customer identifier and customer-level aggregation, these are record segments, not necessarily customer segments.

Model results depend on data quality, row count, feature choice and target leakage. Review metrics and business context before relying on predictions.

## Security and current deployment limits

- Each browser receives a random session token; API routes reject missing, unknown or expired session IDs. No shared `default` dataset session exists.
- API responses include no-store and basic security headers. SQL runs in a separate in-memory SQLite connection with a read-only authorizer.
- CORS is restricted to configured origins. Set production origins explicitly.
- Uploaded datasets are held in backend process memory for up to two hours by default and are lost when the process restarts. This version is intended for a single backend instance / one worker. For multiple workers, replicas or durable production use, move session data to Redis or a database and add user authentication, rate limiting, monitoring and encrypted storage.
- The frontend also keeps its active workspace in browser localStorage for refresh recovery. Clear the workspace and browser storage when using a shared computer.
- The original Design Arena page-view and session-recording scripts have been removed from this project package.

## Tests

Install the requirements, then run:

```bash
python -m pytest -q backend/tests
python -m py_compile backend/app.py
npm run check:js
```

Automated API tests cover session isolation, read-only SQL and all three ML endpoints.

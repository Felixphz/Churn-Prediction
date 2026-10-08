# AGENTS.md — Churn Prediction Platform

## Layout (non-obvious)
- `backend/` — FastAPI app. Entrypoint: `backend/app/main.py`. Routers: `app/routers/` (all prefixed `/api/v1/*`; retrain routes live in `pipeline.py` under `/api/v1/retrain*`).
- `backend/scripts/` — `retrain.py` (monthly GridSearchCV pipeline) and `seed_customers.py`. `backend/app/utils/` holds the real inference code (`preprocessor.py`, `model_predictor.py`, `batch_predictor.py`, `scheduler.py`).
- `ml_pipeline/` — notebooks (`eda`, `data_preprocessing`, `model_training`) + `src/data_ingestion.py` only. No training scripts here; production training is `backend/scripts/retrain.py`.
- `dashboard/` — Streamlit `app.py`. `infra/` — Dockerfiles for postgres + mlflow. Spec: `docs/BACKEND_REQUIREMENTS.md` (drifts from code; code wins).

## Run / verify
- Canonical: `docker compose up --build` (postgres `:5432`, api `:8000`, dashboard `:8501`, mlflow host `${MLFLOW_PORT:-5001}` → container `:5000`).
- Container boot order is enforced: `backend/entrypoint.sh` runs `alembic upgrade head` → `python scripts/seed_customers.py` (skipped when `SEED_ON_STARTUP=false`) → `uvicorn app.main:app`. Don't replace with bare `uvicorn` in Docker.
- Local API (no Docker): run Postgres, copy `backend/.env.example` → `backend/.env` (it uses `localhost`; root `.env.example` uses Docker hostnames `postgres`/`mlflow`), then from `backend/`: `alembic upgrade head && uvicorn app.main:app --port 8000`.
- New migration from `backend/`: `alembic revision --autogenerate -m "<msg>"` (`alembic/env.py` reads `DATABASE_URL` from settings and imports all models).
- No tests, lint, typecheck, or CI exist (`pytest`/`httpx` are in `backend/requirements.txt` but there are zero test files).

## Gotchas — read before touching ML/DB code
- `*.csv`, `*.joblib`, `*.pkl` are gitignored. Three git-absent files break things silently:
  1. `ml_pipeline/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` — generate via `python ml_pipeline/src/data_ingestion.py` (kagglehub download); `seed_customers.py` fails without it.
  2. Model loading is **MLflow Registry-based**: `ModelLoader.get_model()` loads `models:/${REGISTRY_MODEL_NAME}@${MODEL_ALIAS}` (defaults `churn-model@production`) via `MLFLOW_TRACKING_URI` — no local joblib. Nothing registered = 500 with a clear error. Bootstrap once per environment: `python backend/scripts/register_initial_model.py`. MLflow server must be reachable at first predict (lazy load, cached in memory; auto-reloads within `MODEL_REFRESH_TTL_SECONDS`=30 when the alias points to a new version — a failed alias check keeps serving the cache). `retrain.py` registers + promotes the alias when improved. SHAP picks an explainer by model type (`TreeExplainer` → `LinearExplainer` → `PermutationExplainer`); if none works, prediction still returns with `shap_available: false` instead of a 500.
  3. `ml_pipeline/data/resampled/smote_enn/{X_train,y_train}.csv` + `original/{X_test,y_test}.csv` — notebook outputs; `run_retrain()` fails without them. MLflow logging failure is caught and non-fatal; missing CSVs are not.
- Preprocessor contract (`backend/app/utils/preprocessor.py`): replicates `pd.get_dummies(drop_first=True)` from `data_preprocessing.ipynb` → exactly 30 features in `EXPECTED_FEATURE_NAMES`. Keep in sync with the notebook; never re-derive encoding independently.
- Case mismatch is intentional: DB/schemas use `snake_case` (`senior_citizen`), preprocessor expects notebook `CapitalCase` keys (`SeniorCitizen`, `MonthlyCharges`, …). Follow the mapping in `batch_predictor.py` when adding callers.
- Threshold default is `0.30`, not `0.5` (`MODEL_THRESHOLD`, `DEFAULT_MODEL=GradientBoosting`). Retrain only promotes when F1 exceeds active model by `RETRAIN_MIN_IMPROVEMENT` (default `0.01`).
- Scheduler is in-process APScheduler (`app/utils/scheduler.py`, cron day=1 02:00), disabled when `ENVIRONMENT=test` or `ENABLE_SCHEDULER=false`. Manual run: `POST /api/v1/retrain/trigger`.
- `dashboard/app.py` reads `API_URL` from env (default `http://api:8000` for Docker); local/Streamlit Cloud must set it. Timeouts: GET 60s, predict 90s, batch 180s, retrain 90s.

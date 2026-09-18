# Backend Requirements - Churn Prediction Platform

## 1. Visión General

Plataforma de predicción de churn para clientes de telecomunicaciones. El sistema almacena información de clientes, ejecuta predicciones mensuales (batch) y bajo demanda, proporciona explicabilidad SHAP por predicción, y reentrena el modelo mensualmente con datos actualizados.

### Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  PostgreSQL   │  │  MLflow      │  │  API REST    │      │
│  │  (datos +     │  │  Server      │  │  (FastAPI)   │      │
│  │  resultados)  │  │  (tracking)  │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │                │
│  ┌──────┴─────────────────┴─────────────────┴──────┐        │
│  │              Pipeline (retrain.py)               │        │
│  │  - Preprocesamiento de datos                     │        │
│  │  - Entrenamiento de modelos                      │        │
│  │  - Comparación y selección del mejor modelo      │        │
│  │  - Predicciones batch mensuales                  │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  ┌──────────────┐                                          │
│  │  Dashboard   │  ← Streamlit (futuro)                    │
│  │  (Streamlit) │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Usuario     │
                   │  (Analista)  │
                   └──────────────┘
```

---

## 2. Base de Datos (PostgreSQL)

### 2.1 Tabla `customers`

Almacena la información de cada cliente. Las columnas coinciden con las features del dataset de entrenamiento (IBM Telco Customer Churn).

```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    -- Identificadores
    customer_id VARCHAR(50) UNIQUE NOT NULL,  -- ID original del dataset

    -- Datos demográficos
    gender VARCHAR(10) NOT NULL,              -- Male/Female
    senior_citizen BOOLEAN NOT NULL,          -- 0/1
    partner VARCHAR(5) NOT NULL,              -- Yes/No
    dependents VARCHAR(5) NOT NULL,           -- Yes/No

    -- Servicios contratados
    tenure INTEGER NOT NULL,                  -- Meses como cliente (0-72)
    phone_service VARCHAR(5) NOT NULL,        -- Yes/No
    multiple_lines VARCHAR(20),               -- Yes/No/No phone service
    internet_service VARCHAR(20) NOT NULL,    -- DSL/Fiber optic/No
    online_security VARCHAR(20),              -- Yes/No/No internet service
    online_backup VARCHAR(20),                -- Yes/No/No internet service
    device_protection VARCHAR(20),            -- Yes/No/No internet service
    tech_support VARCHAR(20),                 -- Yes/No/No internet service
    streaming_tv VARCHAR(20),                 -- Yes/No/No internet service
    streaming_movies VARCHAR(20),             -- Yes/No/No internet service

    -- Contrato y facturación
    contract VARCHAR(20) NOT NULL,            -- Month-to-month/One year/Two year
    paperless_billing VARCHAR(5) NOT NULL,    -- Yes/No
    payment_method VARCHAR(30) NOT NULL,      -- Electronic check/Mailed check/etc.
    monthly_charges NUMERIC(10,2) NOT NULL,
    total_charges NUMERIC(10,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_customers_customer_id ON customers(customer_id);
```

### 2.2 Tabla `predictions`

Almacena cada predicción realizada, incluyendo el resultado y la explicabilidad SHAP.

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,          -- Nombre del modelo usado
    model_version VARCHAR(50),                -- Versión del modelo (MLflow run_id)
    threshold NUMERIC(5,4) NOT NULL,          -- Umbral de decisión usado

    -- Predicción
    churn_probability NUMERIC(5,4) NOT NULL,  -- Probabilidad de churn (0-1)
    churn_prediction BOOLEAN NOT NULL,        -- True = churn, False = no churn

    -- Explicabilidad SHAP (valores por feature en formato JSONB)
    shap_values JSONB NOT NULL,               -- {"tenure": -0.15, "monthly_charges": 0.23, ...}
    shap_base_value NUMERIC(8,6),             -- Valor base del modelo
    top_features JSONB,                       -- Top N features más influyentes

    -- Metadata
    prediction_type VARCHAR(20) NOT NULL,     -- 'batch' | 'on_demand'
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Para batch mensual
    batch_run_id VARCHAR(50)                  -- ID del lote de predicción batch
);

CREATE INDEX idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at);
CREATE INDEX idx_predictions_batch_run_id ON predictions(batch_run_id);
CREATE INDEX idx_predictions_model_name ON predictions(model_name);
```

### 2.3 Tabla `model_registry`

Registro de modelos activos y su historial.

```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,       -- MLflow run_id
    mlflow_experiment_id VARCHAR(50),
    mlflow_run_uri VARCHAR(200),

    -- Métricas del modelo
    f1_score NUMERIC(6,4),
    recall NUMERIC(6,4),
    auc_roc NUMERIC(6,4),
    precision_score NUMERIC(6,4),

    -- Configuración
    resampling_strategy VARCHAR(30),          -- smote_enn, smote, etc.
    hyperparameters JSONB,                    -- Hiperparámetros del modelo
    threshold NUMERIC(5,4),                   -- Umbral óptimo seleccionado
    feature_names JSONB,                      -- Lista de features esperadas

    -- Estado
    is_active BOOLEAN DEFAULT FALSE,          -- Solo un modelo activo a la vez
    activated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_model_registry_active ON model_registry(is_active);
```

### 2.4 Tabla `retrain_history`

Historial de ejecuciones del pipeline de reentrenamiento.

```sql
CREATE TABLE retrain_history (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    status VARCHAR(20) NOT NULL,              -- 'running' | 'success' | 'failed'

    -- Datos de entrenamiento
    total_customers INTEGER,
    train_samples INTEGER,
    test_samples INTEGER,

    -- Resultados
    models_evaluated INTEGER,
    best_model_name VARCHAR(50),
    best_model_mlflow_run_id VARCHAR(50),
    best_f1_score NUMERIC(6,4),
    best_recall NUMERIC(6,4),
    best_auc_roc NUMERIC(6,4),
    model_improved BOOLEAN DEFAULT FALSE,     -- ¿Mejoró respecto al modelo activo?

    -- Errores
    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.5 Tabla `api_logs` (opcional, para auditoría)

```sql
CREATE TABLE api_logs (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    customer_id INTEGER,
    response_status INTEGER,
    response_time_ms INTEGER,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. API REST (FastAPI)

### 3.1 Endpoints de Clientes

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/customers` | Listar clientes (paginación, filtros) |
| `GET` | `/api/v1/customers/{customer_id}` | Obtener información de un cliente |
| `POST` | `/api/v1/customers` | Crear/actualizar cliente |
| `POST` | `/api/v1/customers/bulk` | Carga masiva de clientes (CSV/JSON) |

### 3.2 Endpoints de Predicciones

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/predictions/{customer_id}` | Obtener última predicción de un cliente |
| `GET` | `/api/v1/predictions/{customer_id}/history` | Historial de predicciones de un cliente |
| `POST` | `/api/v1/predictions/predict` | Ejecutar predicción on-demand para un cliente |
| `POST` | `/api/v1/predictions/batch` | Ejecutar predicciones batch para todos los clientes |
| `GET` | `/api/v1/predictions/batch/{run_id}` | Estado de un lote de predicciones batch |

### 3.3 Endpoints de Explicabilidad

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/explain/{prediction_id}` | Obtener explicación SHAP de una predicción |
| `GET` | `/api/v1/explain/global` | Importancia global de features (across all predictions) |

### 3.4 Endpoints de Modelos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/models` | Listar modelos registrados |
| `GET` | `/api/v1/models/active` | Obtener modelo activo actual |
| `POST` | `/api/v1/models/{model_id}/activate` | Activar un modelo específico |
| `GET` | `/api/v1/models/{model_id}/metrics` | Métricas de un modelo |

### 3.5 Endpoints de Pipeline

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/retrain/status` | Estado del último reentrenamiento |
| `GET` | `/api/v1/retrain/history` | Historial de reentrenamientos |
| `POST` | `/api/v1/retrain/trigger` | Disparar reentrenamiento manual (admin) |

### 3.6 Endpoints de Dashboard

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/dashboard/summary` | Resumen: total clientes, churn rate, métricas |
| `GET` | `/api/v1/dashboard/churn-by-segment` | Churn por segmento (contrato, servicio, etc.) |
| `GET` | `/api/v1/dashboard/timeline` | Evolución de churn en el tiempo |
| `GET` | `/api/v1/dashboard/global-explainability` | SHAP global agregado |

### 3.7 Ejemplo de Respuesta: Predicción con Explicabilidad

```json
{
  "customer_id": "7590-VHVEG",
  "prediction": {
    "churn_probability": 0.8234,
    "churn_predicted": true,
    "threshold": 0.30,
    "model": "GradientBoosting",
    "model_version": "run_abc123"
  },
  "explainability": {
    "top_features": [
      {"feature": "contract_Month-to-month", "shap_value": 0.312, "direction": "increases churn"},
      {"feature": "tenure", "shap_value": -0.187, "direction": "decreases churn"},
      {"feature": "internet_service_Fiber optic", "shap_value": 0.156, "direction": "increases churn"},
      {"feature": "payment_method_Electronic check", "shap_value": 0.098, "direction": "increases churn"},
      {"feature": "monthly_charges", "shap_value": 0.076, "direction": "increases churn"}
    ],
    "base_value": 0.2654,
    "all_shap_values": {"contract_Month-to-month": 0.312, "tenure": -0.187, "...": "..."}
  },
  "predicted_at": "2026-09-17T10:30:00Z",
  "prediction_type": "on_demand"
}
```

---

## 4. Pipeline ML

### 4.1 `ml_pipeline/src/data_preprocessing.py`

Módulo reutilizable que transforma datos crudos de cliente al formato esperado por el modelo.

**Responsabilidades:**
- One-Hot Encoding de variables categóricas (mismo encoder usado en entrenamiento)
- Validación de esquema de entrada
- Manejo de valores faltantes
- Escalado si es necesario

**Input:** Diccionario/JSON con datos crudos del cliente
**Output:** Array numpy/lista con features en el orden correcto para el modelo

**Nota crítica:** El encoder (OneHotEncoder) y el orden de features DEBEN ser idénticos a los usados en el entrenamiento original. Se debe serializar y guardar junto al modelo.

### 4.2 `ml_pipeline/src/predict.py`

Módulo de inferencia que carga el modelo activo y genera predicciones + SHAP values.

**Responsabilidades:**
- Cargar modelo activo desde `model_registry`
- Ejecutar preprocesamiento con `data_preprocessing.py`
- Generar predicción (probabilidad + clase)
- Calcular SHAP values para la predicción
- Retornar resultado estructurado

**SHAP:**
- Usar `shap.TreeExplainer` para modelos basados en árboles (GradientBoosting, XGBoost, RandomForest)
- Calcular SHAP values por instancia
- Extraer top N features más influyentes

### 4.3 `ml_pipeline/src/retrain.py`

Pipeline mensual de reentrenamiento.

**Flujo:**
1. Cargar datos actuales de `customers` (PostgreSQL)
2. Aplicar preprocesamiento (encoding, split)
3. Entrenar modelos candidatos (mismos 5 modelos del notebook)
4. Evaluar con métricas (F1, Recall, AUC-ROC)
5. Comparar mejor modelo vs modelo activo actual
6. Si mejora → registrar en MLflow, actualizar `model_registry`, activar nuevo modelo
7. Registrar ejecución en `retrain_history`
8. (Opcional) Re-ejecutar predicciones batch con nuevo modelo

**Comparación de modelos:**
- Solo reemplazar si el nuevo modelo mejora al actual por un margen mínimo (ej: +1% F1)
- Evitar reentrenamientos que degraden el modelo por overfitting a datos nuevos

### 4.4 `ml_pipeline/src/batch_predict.py`

Ejecuta predicciones batch sobre todos los clientes activos.

**Flujo:**
1. Cargar todos los clientes de PostgreSQL
2. Ejecutar predicción + SHAP para cada uno
3. Guardar resultados en `predictions`
4. Retornar estadísticas del lote (total procesados, churn rate estimado)

---

## 5. MLflow Server

### 5.1 Configuración

- **Servidor MLflow** dockerizado con PostgreSQL como backend de metadata
- **Artifacts** almacenados en volume compartido o S3 (decidir según hosting)
- **Tracking URI:** `http://mlflow-server:5000`

### 5.2 Qué se trackea

| Elemento | Detalle |
|----------|---------|
| **Experimentos** | `churn_prediction` (ya existente con 65 runs) |
| **Runs** | Cada ejecución de entrenamiento y reentrenamiento |
| **Métricas** | F1-Macro, Recall, AUC-ROC, Precision |
| **Parámetros** | Modelo, dataset, hiperparámetros, threshold, resampling strategy |
| **Artifacts** | Modelo serializado (joblib/pickle), encoder, metadata |
| **Model Registry** | Modelos versionados con stages (Staging → Production) |

### 5.3 Integración con `model_registry`

- Al activar un modelo, se registra su `mlflow_run_id` en `model_registry`
- La API carga el modelo desde MLflow usando el `run_id`
- Mantener historial de todos los modelos evaluados

---

## 6. Cron Job - Reentrenamiento Mensual

### 6.1 Configuración

```yaml
# Schedule: primer día de cada mes a las 02:00 UTC
schedule: "0 2 1 * *"
command: "python ml_pipeline/src/retrain.py"
```

### 6.2 Flujo del Cron Job

```
1. Verificar conectividad con PostgreSQL y MLflow
2. Ejecutar retrain.py:
   a. Extraer todos los clientes de la tabla customers
   b. Preprocesar datos
   c. Entrenar 5 modelos con GridSearchCV
   d. Evaluar y comparar con modelo activo
   e. Si hay mejora → activar nuevo modelo
   f. Registrar en retrain_history
3. Ejecutar batch_predict.py (opcional, re-predecir con nuevo modelo)
4. Notificar resultado (log/email futura integración)
```

### 6.3 Implementación

Opciones para el cron job:
- **Docker:** Container dedicado con `crond` + script Python
- **APScheduler:** Dentro del container de la API con scheduler integrado
- **Render/Railway:** Cron jobs nativos del platform (si están disponibles)

**Recomendación:** Usar APScheduler dentro del container de la API para simplificar la infraestructura.

---

## 7. Dashboard (Streamlit) - Fase Futura

### 7.1 Requisitos Mínimos (MVP)

- Consulta de cliente por ID → mostrar predicción + top SHAP features
- Tabla de clientes con columna de churn probability
- Filtros por contract type, internet service, tenure

### 7.2 Features Futuros

- Dashboard en vivo con actualización automática (streaming o polling)
- Visualización global de SHAP (summary plot, dependence plots)
- Métricas del modelo en tiempo real
- Alertas de cambios en patrones de churn

---

## 8. Infraestructura (Docker Compose)

### 8.1 Servicios

```yaml
services:
  postgres:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: churn_prediction
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.15.0
    command: mlflow server --backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    ports:
      - "5000:5000"
    depends_on:
      - postgres

  api:
    build: ./backend
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/churn_prediction
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - mlflow

  scheduler:
    build: ./backend
    command: python ml_pipeline/src/retrain.py
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/churn_prediction
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - postgres
      - mlflow
    # Cron schedule se maneja con APScheduler dentro del container

  dashboard:
    build: ./dashboard
    command: streamlit run app.py --server.port 8501
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  pgdata:
  mlflow_artifacts:
```

---

## 9. Estructura de Directorios (Backend)

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, startup events, CORS
│   ├── config.py            # Settings (env vars, database URL, etc.)
│   ├── database.py          # SQLAlchemy engine, session, Base
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── customer.py
│   │   ├── prediction.py
│   │   ├── model_registry.py
│   │   └── retrain_history.py
│   ├── schemas/             # Pydantic schemas (request/response)
│   │   ├── __init__.py
│   │   ├── customer.py
│   │   ├── prediction.py
│   │   └── model.py
│   ├── routers/             # API routes
│   │   ├── __init__.py
│   │   ├── customers.py
│   │   ├── predictions.py
│   │   ├── explainability.py
│   │   ├── models.py
│   │   ├── retrain.py
│   │   └── dashboard.py
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   ├── prediction_service.py
│   │   ├── explainability_service.py
│   │   ├── model_service.py
│   │   └── retrain_service.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py  # Data preprocessing pipeline
├── ml_pipeline/
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── predict.py
│   │   ├── retrain.py
│   │   └── batch_predict.py
│   └── data/
│       └── results/
│           └── best_model_GradientBoosting.joblib
├── requirements.txt
├── Dockerfile
└── alembic/                 # Migraciones de BD (opcional)
    └── versions/
```

---

## 10. Dependencias Adicionales (requirements.txt backend)

```
# API
fastapi
uvicorn[standard]
pydantic>=2.0

# Database
sqlalchemy>=2.0
psycopg2-binary
alembic

# ML
scikit-learn
xgboost
shap
joblib

# MLflow
mlflow>=2.10

# Utils
python-dotenv
pandas
numpy

# Scheduler
apscheduler

# Testing
pytest
httpx
```

---

## 11. Variables de Entorno

```env
# Database
DATABASE_URL=postgresql://user:password@postgres:5432/churn_prediction
DB_USER=user
DB_PASSWORD=password

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# Model
MODEL_THRESHOLD=0.30
DEFAULT_MODEL=GradientBoosting

# Retrain
RETRAIN_MIN_IMPROVEMENT=0.01  # 1% mínimo de mejora para activar nuevo modelo
```

---

## 12. Secuencia de Implementación (Orden Recomendado)

1. **Fase 1: Fundamentos**
   - Configurar estructura del backend (`backend/app/`)
   - Configurar SQLAlchemy + modelos ORM
   - Crear migraciones iniciales de BD
   - Configurar FastAPI con health check

2. **Fase 2: Data Layer**
   - Implementar endpoints CRUD de clientes
   - Script de seed data (migrar datos del dataset a PostgreSQL)
   - Validar esquema de datos

3. **Fase 3: Predicción**
   - Adaptar `data_preprocessing.py` como módulo reutilizable
   - Implementar `predict.py` con SHAP
   - Endpoint de predicción on-demand
   - Registrar modelo activo en `model_registry`

4. **Fase 4: Batch + Cron**
   - Implementar `batch_predict.py`
   - Configurar APScheduler para reentrenamiento mensual
   - Implementar `retrain.py` como script CLI
   - Tabla `retrain_history`

5. **Fase 5: MLflow Server**
   - Dockerizar MLflow con PostgreSQL backend
   - Migrar experimentos existentes
   - Integrar tracking en retrain pipeline

6. **Fase 6: Docker Compose**
   - Crear Dockerfiles para cada servicio
   - Configurar docker-compose.yml completo
   - Variables de entorno y secrets

7. **Fase 7: Dashboard (Futuro)**
   - Streamlit MVP con consulta de clientes
   - Visualización de SHAP values
   - Conexión con API REST

---

## 13. Decisiones Pendientes

| # | Decisión | Opciones | Estado |
|---|----------|----------|--------|
| 1 | Hosting final | Render / Railway / VPS | Pendiente |
| 2 | MLflow artifacts storage | Volume local / S3 / GCS | Pendiente |
| 3 | Autenticación API | Sin auth / API Key / JWT | Pendiente (¿necesaria?) |
| 4 | Migraciones BD | Alembic / SQL manual | Pendiente |
| 5 | Logging | Print / Loguru / structlog | Pendiente |
| 6 | CI/CD | GitHub Actions / manual | Pendiente |

---

## 14. Notas Importantes

### Compatibilidad del Modelo
- El `OneHotEncoder` y el orden de features DEBEN ser idénticos a los usados en el notebook `model_training.ipynb`
- Guardar el encoder serializado (joblib) junto al modelo
- El pipeline de preprocesamiento en producción debe replicar exactamente el del notebook

### SHAP Values
- Para GradientBoosting (modelo actual), usar `shap.TreeExplainer`
- Los SHAP values se almacenan como JSONB para flexibilidad
- Considerar comprimir valores (top 10 features) para ahorrar espacio

### Performance
- Predicción on-demand: < 200ms target
- Batch mensual: puede tomar varios minutos (aceptable)
- SHAP: puede ser costoso computacionalmente → cachear resultados frecuentes

### Seguridad
- No exponer datos sensibles de clientes en logs
- Rate limiting en endpoints públicos (si aplica)
- Validación estricta de inputs en la API

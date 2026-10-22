from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import customers, predictions, pipeline, dashboard, outcomes

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting Churn Prediction API - {settings.ENVIRONMENT}")
    # The monthly retrain cron runs in the separate retrainer service
    yield
    print("Shutting down Churn Prediction API")


app = FastAPI(
    title="Churn Prediction API",
    description="API para predicción de churn de clientes de telecomunicaciones",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customers.router)
app.include_router(predictions.router)
app.include_router(pipeline.router)
app.include_router(dashboard.router)
app.include_router(outcomes.router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }

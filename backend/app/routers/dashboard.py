from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from app.schemas.pipeline import DashboardSummary

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)):
    total_customers = db.query(func.count(Customer.id)).scalar() or 0
    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0

    churn_count = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.churn_prediction == True)
        .scalar()
        or 0
    )

    active_model = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()

    last_retrain = (
        db.query(RetrainHistory)
        .order_by(RetrainHistory.started_at.desc())
        .first()
    )

    churn_rate = round(churn_count / total_customers * 100, 2) if total_customers > 0 else 0

    return DashboardSummary(
        total_customers=total_customers,
        churn_count=churn_count,
        churn_rate=churn_rate,
        active_model=active_model.model_name if active_model else None,
        total_predictions=total_predictions,
        last_retrain=last_retrain.started_at.isoformat() if last_retrain else None,
    )


# ---------------------------------------------------------------------------
# Analytics para el dashboard de Streamlit
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

SERVICE_COLUMNS = {
    "phone_service": "Teléfono",
    "multiple_lines": "Múltiples líneas",
    "online_security": "Seguridad online",
    "online_backup": "Backup online",
    "device_protection": "Protección dispositivo",
    "tech_support": "Soporte técnico",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming películas",
}

SEGMENT_COLUMNS = {
    "senior_citizen": "Adulto mayor (65+)",
    "gender": "Género",
    "partner": "Pareja",
    "dependents": "Dependientes",
    "contract": "Contrato",
    "internet_service": "Internet",
    "payment_method": "Método de pago",
    "paperless_billing": "Factura electrónica",
}


def _num(x):
    return None if x is None else float(x)


def _segment_stats(df: pd.DataFrame, col: str) -> list[dict]:
    g = df.groupby(col, dropna=False)
    out = []
    for value, sub in g:
        scored = sub[sub["churn_prediction"].notna()]
        out.append({
            "value": "Sin dato" if pd.isna(value) else str(value),
            "customers": int(len(sub)),
            "predicted_churn": int(scored["churn_prediction"].astype(bool).sum()),
            "churn_rate": round(float(scored["churn_prediction"].astype(bool).mean() * 100), 2) if len(scored) else None,
            "avg_probability": round(float(scored["churn_probability"].mean()), 4) if len(scored) else None,
            "avg_monthly_charges": round(float(sub["monthly_charges"].mean()), 2),
        })
    return sorted(out, key=lambda r: -r["customers"])


@router.get("/analytics")
def dashboard_analytics(db: Session = Depends(get_db)):
    """Agregados para el dashboard: clientes, servicios, predicciones y modelo."""
    # Última predicción por cliente
    latest = (
        db.query(
            Prediction.customer_id.label("cid"),
            func.max(Prediction.predicted_at).label("last_at"),
        )
        .group_by(Prediction.customer_id)
        .subquery()
    )
    rows = (
        db.query(
            Customer,
            Prediction.churn_probability,
            Prediction.churn_prediction,
        )
        .outerjoin(latest, latest.c.cid == Customer.id)
        .outerjoin(
            Prediction,
            (Prediction.customer_id == latest.c.cid)
            & (Prediction.predicted_at == latest.c.last_at),
        )
        .all()
    )

    records = []
    for c, prob, pred in rows:
        rec = {col: getattr(c, col) for col in list(SERVICE_COLUMNS) + list(SEGMENT_COLUMNS)}
        rec.update(
            customer_id=c.customer_id,
            tenure=c.tenure,
            monthly_charges=_num(c.monthly_charges),
            total_charges=_num(c.total_charges),
            churn_probability=_num(prob),
            churn_prediction=pred,
        )
        records.append(rec)
    df = pd.DataFrame(records)
    if len(df):
        df = df.drop_duplicates(subset="customer_id", keep="first")

    # --- Clientes ---------------------------------------------------------
    customers_block: dict = {"total": int(len(df))}
    if len(df):
        df["senior_citizen"] = df["senior_citizen"].map({True: "Sí", False: "No"})
        df["services_count"] = sum(
            (df[c] == "Yes").astype(int) for c in SERVICE_COLUMNS
        )
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[-1, 12, 24, 48, 60, 1000],
            labels=["0-12 m", "13-24 m", "25-48 m", "49-60 m", "60+ m"],
        ).astype(str)

        customers_block.update(
            avg_tenure=round(float(df["tenure"].mean()), 1),
            avg_monthly_charges=round(float(df["monthly_charges"].mean()), 2),
            total_monthly_revenue=round(float(df["monthly_charges"].sum()), 2),
            senior_pct=round(float((df["senior_citizen"] == "Sí").mean() * 100), 2),
            avg_services=round(float(df["services_count"].mean()), 2),
            segments={
                label: _segment_stats(df, col) for col, label in SEGMENT_COLUMNS.items()
            },
            tenure_groups=_segment_stats(df, "tenure_group"),
            services_count=_segment_stats(df, "services_count"),
            services=[
                {
                    "service": label,
                    "adoption_pct": round(float((df[col] == "Yes").mean() * 100), 2),
                    "customers": int((df[col] == "Yes").sum()),
                    "churn_rate_with": (
                        round(float(df.loc[(df[col] == "Yes") & df["churn_prediction"].notna(), "churn_prediction"].astype(bool).mean() * 100), 2)
                        if ((df[col] == "Yes") & df["churn_prediction"].notna()).any() else None
                    ),
                    "churn_rate_without": (
                        round(float(df.loc[(df[col] != "Yes") & df["churn_prediction"].notna(), "churn_prediction"].astype(bool).mean() * 100), 2)
                        if ((df[col] != "Yes") & df["churn_prediction"].notna()).any() else None
                    ),
                }
                for col, label in SERVICE_COLUMNS.items()
            ],
            # Muestra para histogramas / dispersión (sin datos sensibles)
            sample=df[["tenure", "monthly_charges", "churn_probability", "contract",
                       "senior_citizen", "services_count"]]
            .sample(min(len(df), 2000), random_state=42)
            .astype(object).where(lambda d: d.notna(), None)
            .to_dict(orient="records"),
        )

    # --- Predicciones -----------------------------------------------------
    preds = pd.DataFrame(
        db.query(
            Prediction.predicted_at,
            Prediction.churn_probability,
            Prediction.churn_prediction,
            Prediction.prediction_type,
            Prediction.model_name,
            Prediction.top_features,
        ).all(),
        columns=["predicted_at", "prob", "pred", "type", "model", "top_features"],
    )
    predictions_block: dict = {"total": int(len(preds))}
    if len(preds):
        preds["prob"] = preds["prob"].astype(float)
        preds["day"] = pd.to_datetime(preds["predicted_at"]).dt.date.astype(str)
        daily = preds.groupby("day").agg(
            predictions=("pred", "size"),
            churn=("pred", "sum"),
            avg_probability=("prob", "mean"),
        ).reset_index()
        bins = [0, 0.3, 0.6, 1.0001]
        risk = pd.cut(df["churn_probability"].dropna(), bins=bins, right=False,
                      labels=["Bajo (<30%)", "Medio (30-60%)", "Alto (≥60%)"])
        # Importancia global: |SHAP| medio sobre las top features de cada predicción
        feat_abs: dict = {}
        feat_n: dict = {}
        for tf in preds["top_features"].dropna():
            for it in tf if isinstance(tf, list) else []:
                if isinstance(it, dict) and it.get("feature"):
                    name = it["feature"]
                    feat_abs[name] = feat_abs.get(name, 0.0) + abs(float(it.get("shap_value") or 0))
                    feat_n[name] = feat_n.get(name, 0) + 1
        n_preds = max(len(preds), 1)

        predictions_block.update(
            customers_scored=int(df["churn_probability"].notna().sum()),
            predicted_churn=int(df["churn_prediction"].fillna(False).astype(bool).sum()),
            avg_probability=round(float(df["churn_probability"].dropna().mean()), 4),
            by_type=preds["type"].value_counts().to_dict(),
            by_model=preds["model"].value_counts().to_dict(),
            risk_levels={str(k): int(v) for k, v in risk.value_counts().sort_index().items()},
            daily=daily.round(4).to_dict(orient="records"),
            top_features=sorted(
                [
                    {"feature": k, "mean_abs_shap": round(v / n_preds, 5), "count": feat_n[k]}
                    for k, v in feat_abs.items()
                ],
                key=lambda r: -r["mean_abs_shap"],
            )[:15],
            probability_histogram=[
                {"bin": f"{i*10}-{i*10+10}%", "count": int(n)}
                for i, n in enumerate(
                    pd.cut(df["churn_probability"].dropna(),
                           bins=[x / 10 for x in range(11)], include_lowest=True)
                    .value_counts().sort_index().values
                )
            ],
        )

    # --- Modelo -----------------------------------------------------------
    def _model(m: ModelRegistry) -> dict:
        return {
            "model_name": m.model_name,
            "model_version": m.model_version,
            "f1_score": _num(m.f1_score),
            "recall": _num(m.recall),
            "precision": _num(m.precision_score),
            "auc_roc": _num(m.auc_roc),
            "threshold": _num(m.threshold),
            "resampling_strategy": m.resampling_strategy,
            "is_active": bool(m.is_active),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }

    models = db.query(ModelRegistry).order_by(ModelRegistry.created_at).all()
    active = next((m for m in models if m.is_active), None)
    retrains = db.query(RetrainHistory).order_by(RetrainHistory.started_at).all()

    model_block = {
        "active": _model(active) if active else None,
        "registry": [_model(m) for m in models],
        "retrain_history": [
            {
                "run_id": r.run_id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "status": r.status,
                "best_model_name": r.best_model_name,
                "f1_score": _num(r.best_f1_score),
                "recall": _num(r.best_recall),
                "auc_roc": _num(r.best_auc_roc),
                "model_improved": bool(r.model_improved),
            }
            for r in retrains
        ],
    }

    return {
        "customers": customers_block,
        "predictions": predictions_block,
        "model": model_block,
    }

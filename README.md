# Churn Prediction Platform

Proyecto Integrador I: plataforma para predecir el abandono de clientes (churn) usando el dataset IBM Telco Customer Churn. Compara varios modelos de aprendizaje automático y expone las predicciones mediante una API y un dashboard.

> ⚠️ Proyecto en etapa inicial. Aún no hay código implementado.

## Estructura

```
churn-prediction/
├── ml_pipeline/   # EDA, entrenamiento y comparación de modelos
├── backend/       # API REST + base de datos
├── dashboard/     # Visualización de predicciones
├── infra/         # Docker y despliegue
└── docs/          # Documentación
```

## Stack pensado

- **ML**: Python, scikit-learn, XGBoost
- **Tracking de experimentos**: MLflow
- **Backend**: FastAPI + PostgreSQL
- **Dashboard**: por definir
- **Despliegue**: Docker + Render/Railway

## Próximos pasos

- [ ] Preparar y limpiar el dataset
- [ ] Configurar MLflow para trackear experimentos
- [ ] Entrenar primeros modelos base

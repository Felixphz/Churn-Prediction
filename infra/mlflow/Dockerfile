FROM ghcr.io/mlflow/mlflow:v3.4.0

LABEL org.opencontainers.image.source=https://github.com/felixphz/Churn-Prediction

RUN pip install --no-cache-dir psycopg2-binary boto3

ENV MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow-artifacts \
    AWS_DEFAULT_REGION=us-east-2 \
    AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
    AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
    MLFLOW_SQLALCHEMYSTORE_POOL_SIZE=2 \
    MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW=2

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:?Falta MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2
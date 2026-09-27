FROM ghcr.io/mlflow/mlflow:v2.21.0

RUN pip install --no-cache-dir psycopg2-binary boto3

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2
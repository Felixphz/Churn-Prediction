FROM ghcr.io/mlflow/mlflow:v3.4.0

LABEL org.opencontainers.image.source=https://github.com/felixphz/Churn-Prediction

# azure-storage-blob/azure-identity: artifact store on Azure Blob (wasbs://)
RUN pip install --no-cache-dir psycopg2-binary azure-storage-blob azure-identity

# Credentials come from AZURE_STORAGE_CONNECTION_STRING (Container App secret).
# MLFLOW_ARTIFACTS_DESTINATION: wasbs://<container>@<account>.blob.core.windows.net/
ENV MLFLOW_SQLALCHEMYSTORE_POOL_SIZE=2 \
    MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW=2

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:?Falta MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION:?Falta MLFLOW_ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2

FROM ghcr.io/mlflow/mlflow:v2.21.0

# Install PostgreSQL adapter for MLflow backend
RUN pip install psycopg2-binary boto3 azure-storage-blob

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# Default command (overridden by docker-compose)
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

FROM ghcr.io/mlflow/mlflow:v2.21.0

# Install PostgreSQL adapter and S3 client for Supabase
RUN pip install --no-cache-dir psycopg2-binary boto3

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# This command allows the container to start simply.
# In LOCAL: docker-compose overrides this with local DB settings.
# In AZURE: We override this in the portal with Supabase settings.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

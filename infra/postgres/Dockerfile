FROM postgres:16

# Copy init script to create MLflow database
COPY init.sql /docker-entrypoint-initdb.d/init.sql

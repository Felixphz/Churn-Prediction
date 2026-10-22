#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
sleep 3

echo "Running migrations..."
alembic upgrade head

if [ "${SEED_ON_STARTUP:-true}" = "true" ]; then
    echo "Seeding customers..."
    python scripts/seed_customers.py
else
    echo "Skipping seed (SEED_ON_STARTUP=false)"
fi

if [ -n "${TRAINING_DATABASE_URL}" ]; then
    echo "Initializing training DB..."
    python scripts/init_training_db.py
fi

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

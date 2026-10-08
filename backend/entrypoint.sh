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

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

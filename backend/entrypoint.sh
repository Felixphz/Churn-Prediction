#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
sleep 3

echo "Running migrations..."
alembic upgrade head

echo "Seeding customers..."
python scripts/seed_customers.py

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

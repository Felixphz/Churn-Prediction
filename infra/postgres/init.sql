-- This runs on first container start
-- Creates the mlflow database alongside the main churn_prediction DB
CREATE DATABASE mlflow;

-- Labeled outcomes for retraining (TRAINING_DATABASE_URL); in production a separate Neon project
CREATE DATABASE churn_training;

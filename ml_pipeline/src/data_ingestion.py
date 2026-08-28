import kagglehub
import shutil
import os

# Descarga a la caché local (path varía por PC, no lo hardcodees)
download_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Descargado en:", download_path)

# Copiar a una ruta fija y relativa dentro del proyecto
project_raw_dir = os.path.join("ml_pipeline", "data", "raw")
os.makedirs(project_raw_dir, exist_ok=True)

for file in os.listdir(download_path):
    shutil.copy(
        os.path.join(download_path, file),
        os.path.join(project_raw_dir, file)
    )

print("Copiado a:", project_raw_dir)
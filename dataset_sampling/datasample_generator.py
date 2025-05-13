import pandas as pd
import shutil
import os

# Leggere il file CSV
df = pd.read_csv("sample_validation.csv")

# Supponiamo che la prima colonna contenga i nomi dei file
file_names = df.iloc[:, 0].tolist()

# Definire le cartelle
source_folder = r"C:\Users\cancr\OneDrive\Desktop\machine learning\progetto\DATASET_COMPLETO_V2\validation"
destination_folder = "sample_validation_data"

# Creare la cartella di destinazione se non esiste
os.makedirs(destination_folder, exist_ok=True)

# Copiare solo i file specificati
for file_name in file_names:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)
    
    if os.path.exists(source_path):  # Verifica che il file esista
        shutil.copy(source_path, destination_path)

print("Copia completata!")

import pandas as pd
import random

# Leggi il file CSV
file_path = r"c:\Users\antoc\Desktop\repo_github\FillerWordsRemoval\PodcastFillers_validation_labels.csv"
df = pd.read_csv(file_path)

# Mescola casualmente le righe
df_shuffled = df.sample(frac=1, random_state=random.randint(1, 2653)).reset_index(drop=True)

# Salva il risultato in un nuovo file CSV
output_path = r"c:\Users\antoc\Desktop\repo_github\FillerWordsRemoval\PodcastFillers_validation_labels_shuffled.csv"
df_shuffled.to_csv(output_path, index=False)

print(f"File mescolato salvato in: {output_path}")
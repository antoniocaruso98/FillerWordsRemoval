import pandas as pd

# Leggere il file CSV
df = pd.read_csv(r"c:\Users\cancr\OneDrive\Desktop\machine learning\progetto\DATASET_COMPLETO_V2\PodcastFillers_Decentered_validation_shuffled.csv")

# Estrarre 110 dei dati in modo casuale
df_sample = df.sample(frac=0.01)

# Stampare o salvare il risultato
df_sample.to_csv("sample_validation.csv", index=False)

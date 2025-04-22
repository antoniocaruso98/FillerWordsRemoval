import pandas as pd
def check_class_order(csv_files):
    for csv_file in csv_files:
        # Leggi il file CSV
        df = pd.read_csv(csv_file)
        # Ottieni l'ordine delle classi
        class_order = df["label"].unique().tolist()
        print(f"File: {csv_file}")
        print(f"Class order: {class_order}")
        print("-" * 50)

# Specifica i percorsi dei file CSV
csv_files = [
    "PodcastFillers_train_labels_shuffled.csv",
    "PodcastFillers_validation_labels_shuffled.csv",
    "PodcastFillers_test_labels_shuffled.csv"
]

def main():
    # Controlla l'ordine delle classi
    check_class_order(csv_files)

if __name__ == "__main__":
    main()
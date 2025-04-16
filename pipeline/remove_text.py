import os


def elimina_file_txt(cartella_root):
    for percorso_corrente, sottocartelle, file in os.walk(cartella_root):
        for nome_file in file:
            if nome_file.endswith(".txt"):
                percorso_file = os.path.join(percorso_corrente, nome_file)
                try:
                    os.remove(percorso_file)
                    print(f"Eliminato: {percorso_file}")
                except Exception as e:
                    print(f"Errore nell'eliminare {percorso_file}: {e}")


# Inserisci il percorso della cartella root qui
cartella_root = "./train-clean-100"
elimina_file_txt(cartella_root)

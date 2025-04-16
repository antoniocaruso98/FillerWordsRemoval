import os
import shutil


def copia_file_in_flat(cartella_root, cartella_destinazione):
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    for percorso_corrente, sottocartelle, file in os.walk(cartella_root):
        for nome_file in file:
            sorgente_file = os.path.join(percorso_corrente, nome_file)
            destinazione_file = os.path.join(cartella_destinazione, nome_file)

            # Se un file con lo stesso nome esiste già nella destinazione, rinominalo
            contatore = 1
            while os.path.exists(destinazione_file):
                nome_file_base, estensione = os.path.splitext(nome_file)
                nuovo_nome_file = f"{nome_file_base}_{contatore}{estensione}"
                destinazione_file = os.path.join(cartella_destinazione, nuovo_nome_file)
                contatore += 1

            try:
                shutil.copy2(sorgente_file, destinazione_file)
                print(f"Copiato: {sorgente_file} -> {destinazione_file}")
            except Exception as e:
                print(f"Errore nel copiare {sorgente_file}: {e}")


# Specifica il percorso della cartella root e della cartella di destinazione
cartella_root = "./train-clean-100"
cartella_destinazione = "./train-clean-100-flat"

copia_file_in_flat(cartella_root, cartella_destinazione)

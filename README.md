# Note on Dataset and weights

Due to limitations in the file size, uploading either the Dataset and/or the weights on Moodle was not possible.

Thus, we provide two separate Google Drive links to download both of them:

- [DATASET_COMPLETO_V3.zip](https://drive.google.com/file/d/1lpnk34adSk0F-MZSwmQNq1yXLPZuGJD-/view?usp=drive_link)
- [Weights_and_plots.zip](https://drive.google.com/file/d/16DnFEGe9qGahYQg3nC1ohA2-9NHhYYH8/view?usp=sharing)

> Note: in order to make the model work, the correct path to the desired model weights file must be specified in main.py (or, in case, the desired checkpoint.pth file should be copied to the code parent directory). In case of errors when loading the checkpoint.pth file, use the parameter **weights_only = False**.

> Note: the dataset folder should be placed in the code parent folder.

# Model usage

This folder includes 5 python modules:

1. *main.py*: contains the model implementation, training and testing phases
2. *spectrogram.py*: contains some functions used to load and preprocess audio
3. *validation.py*: contains code which enables to perform validation phase only
4. *ResNet8.py*: contains a Pytorch implementation of ResNet8 model, which is not available in torchvision package
5. *inference.py*: contains code which enables to used the trained model in order to produce clean audio files



REQUIREMENTS
------------------

Si vuole adattare una soluzione che si occupa di rimuovere intercalari/silenzi/versi da tracce audio, al fine di:
1. Estenderne il funzionamento anche al video
2. (Eventualmente, in sostituzione di 1.) Cercare di modificare la struttura della rete descritta dal paper di riferimento, lavorando esclusivamente sulla parte audio.

Non avendo al momento chiaro quale delle due scelte adottare, è stato individuato esclusivamente un dataset che permette di addestrare la sola parte audio (di cui è fornito il link), mentre non siamo ancora a conoscenza di possibili dataset che potrebbero essere utilizzati per l'addestramento della parte video. Riguardo il dataset disponibile per la parte audio, esso contiene (ad alto livello) informazioni su:
- label dell'evento (es. uhm, silenzio, filler word...)
- istante di inizio dell'intervallo contenente la keyword
- istante di fine dell'intervallo contenente la keyword

Il punto di partenza per la realizzazione del progetto è costituito dal paper "Filler Word Detection and Classification: A Dataset and Benchmark, 2022", che descrive una architettura a pipeline con diversi stadi: [ASR] + VAD + Classificatore.

Trattandosi di un task di individuazione di keywords e degli intervalli in cui esse compaiono, quindi sostanzialmente di una sorta di detection, si prevede di valutare le performance tenendo conto della corrispondenza temporale (Intersection over Union) e della correttezza delle classificazioni delle keywords (accuracy, matrici di confusione, precision, recall).


https://podcastfillers.github.io/


Considerando invece l'ipotesi di concentrarci solo sull'audio, gradiremmo dei consigli sui tipi di modifiche che potremmo cercare di apportare rispetto all'implementazione descritta dal paper di riferimento (es. sostituire alcuni moduli interni alla rete, considerare approcci diversi - spettrogramma o end-to-end -, o altro)

OVERLEAF LINK PAPER: https://www.overleaf.com/project/655e2f7c12b597bb6eec9a4c




A dire il vero non so se avessi in mente un articolo preciso,

di sicuro avevo visto un video tutorial su internet, tipo questo (ma non questo): https://www.youtube.com/watch?v=ydsFzu_wPb4

Ho visto che è "uscito" anche un tool per fare ciò: https://www.descript.com/ Si potrebbe prendere spunto da quello

Io pensavo di prendere i video delle lezioni registrate dai docenti del poli, ce ne sono tantissimi, eventualmente anche i miei. Individuare eventuali docenti che abbondano di filler words, oppure se non si trovano "inventarsi qualcosa" tipo un docente che dice sempre "ok?" a fine frase e togliere quello.

L'integrazione con il video può essere anche molto semplice, tipo se nel video c'è qualcosa che si muove (docente che scrive a penna o da tastiera, eventualmente mouse che si muove, ma potrebbe essere inutile, ...) non si può "tagliare", altrimenti sì.



Questo non mi sembra male come lavoro: https://arxiv.org/pdf/2203.15135v1.pdf

Uno strumento utile è avere un riconoscitore del parlato con timestamp a livello di singola parola https://github.com/linto-ai/whisper-timestamped. Praticamente fa già tutto il lavoro perchè basta rimuovere i pezzi marcati come filler. Cosa dobbiamo aggiungere?


Se volete metterla sul ridere: https://github.com/DennisFaucher/FillerWordsShock

Poichè Google Speech-to-Text API trascrive l'audio in tempo reale ma sopprime le filler word, l'autore opta per IBM Watson Speech-to-Text che non le sopprime ma le sostituisce con "%HESITATION". È stato quindi possibile evidenziarle modificando il codice per evidenziare queste occorrenze nel testo trascritto.

Il codice del progetto è stato adattato per: evidenziare le parole di riempimento con simboli visivi (es. “💥”). Modificare la soglia di confidenza per ridurre falsi positivi. Rendere l'interfaccia più semplice e visibile per l'utente.

L'obiettivo generale del progetto è di fornire feedback in tempo reale su queste parole durante presentazioni o discorsi, migliorando così l'efficacia comunicativa


- DATASET ETICHETTATO (PODCASTFILLERS)

https://zenodo.org/records/7121457#.Y0bzKezMKdY

- Filler Word Detection and Classification: A Dataset and Benchmark: https://arxiv.org/pdf/2203.15135v1.pdf
- um_detector: https://github.com/ezxzeng/um_detector 

- Deep Learning Object Detection Approaches to Signal Identification: https://arxiv.org/abs/2210.16173
- You Only Hear Once: A YOLO-like Algorithm for Audio Segmentation and Sound Event Detection: https://arxiv.org/abs/2109.00962
- SpeechYOLO: Detection and Localization of Speech Objects: https://arxiv.org/abs/1904.07704 - repository: https://github.com/MLSpeech/speech_yolo




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

# Descrizione dataset

Il dataset di partenza è quello descritto in uno dei nostri paper di riferimento, [https://arxiv.org/pdf/2203.15135v1] (Filler Word Detection and Classification: A Dataset and Benchmark). In particolare il dataset completo si può trovare al seguente link: [https://podcastfillers.github.io/]

Il dataset in questione è costituito da una serie di **clip audio da 1 secondo** (circa 77.000) tratte da 199 episodi di podcast in lingua inglese, ciascuna **contenente uno e un solo evento** riconducibile a una tra **6 classi di filler words**:

- **Words**: parole confuse per via di disturbo audio, sovrapposizione di voci diverse, ecc.
- **Uh**: intercalare
- **Um**: intercalare
- **Breath**: eventi in cui si sente solo respiro del parlatore
- **Laughter**: eventi in cui si rileva una risata del parlatore
- **Music**: eventi in cui si rileva solo musica strumentale, ma non voce.

Il dataset è già suddiviso in **tre partizioni**, ovvero **train, test e validation**.

Tale dataset, tuttavia, nella sua forma originale, non era sufficiente per i nostri scopi, dunque abbiamo operato una serie di **trasformazioni/aggiunte sia sui dati, sia sulle etichette**:

1. Introduzione di una **nuova classe** che rappresentasse gli **esempi 'negativi'**, ovvero che rispecchiassero il normale parlato in **assenza di filler** words: **classe _Nonfiller_**. Per fare ciò, abbiamo sfruttato la presenza nel dataset, oltre che delle clip con le loro annotazioni, anche degli episodi completi e di annotazioni aggiuntive generate da un VAD (report periodico dell'attività vocale rilevata). Sono quindi state estratte **ulteriori 60K clip**, che hanno la caratteristica di **non avere intersezioni con intervalli contenenti filler words e silenzio** (clip ricercate considerando i soli intervalli in cui il VAD risultava attivo, secondo un'opportuna soglia). Le clip così ottenute sono state **ripartite in proporzione** tra i dataset di train, test e validation.
2. **Shift temporale di tutte le clip appartenenti a classi positive**: nel lavoro originale (di cui sopra), si prevedeva di effettuare esclusivamente classificazione delle clip, che pertanto sono state realizzate in maniera tale che il centro dell'evento da rilevare corrispondesse con 0.5s, ovvero il centro della clip stessa. Nel nostro caso, tuttavia, queste clip così generate si sono rivelate inadatte, poiché oltre alla classificazione, occorreva effettuare anche una **regressione** per prevedere le coordinate temporali del **bounding box** relativo all'evento, espresse in termini  di centro e ampiezza. Occorreva, dunque, effettuare uno shift di ogni clip, in modo tale che le coordinate del centro si potessero trovare, uniformemente, in ogni possibile posizione all'interno della clip, per permettere al modello di **generalizzare correttamente**. Per fare questo, abbiamo sfruttato le annotazioni presenti per rigenerare ogni singola clip positiva estraendola opportunamente dagli episodi completi, applicando un offset casuale (con distribuzione uniforme) rispetto all'intervallo riportato nell'annotazione.
3. **Modifica del formato delle etichette**: inizialmente i timestamp dei vari eventi era riportato nel formato **(start_s, end_s)**, che però era poco appropriato per le nostre esigenze, in quando era necessario poter prevedere anche, eventualmente, bounding box localizzati parzialmente al di fuori della finestra da 1 secondo. Per questo motivo abbiamo scelto di convertire questo formato di timestamp in **(center_t, delta_t)**, cosicché $start\_s = center\_t - delta\_t/2$ e $end\_s = center\_t + delta\_t/2$.

Il raggiungimento di questa formulazione finale del dataset, ha tuttavia richiesto **numerosi tentativi**, molti dei quali fallimentari:

1. Inizialmente, abbiamo pensato di ovviare al **problema delle clip centrate** decentrandole **on-the-fly** durante l'operazione di caricamento dei dati, semplicemente **traslando il dB-MEL-spectrogram** utilizzato come input per la rete convolutiva e inserendo, nella **parte 'mancante'** (ovvero quella rimasta vuota a seguito dello shift), **prima silenzio (riempimento con 0), poi rumore gaussiano**. Tuttavia questa strategia non si è rivelata efficace, perché **non permette al modello di generalizzare** correttamente sui dati forniti in fase di inferenza. Abbiamo quindi optato per rigenerare tutte le clip con offset desiderato.
2. La **presenza di eventuale silenzio** in dati reali, **in fase di inferenza**, costituiva un problema, in quanto nel dataset **non è presente una classe apposita** che descriva questo tipo di evento. Quindi abbiamo pensato di **escludere il silenzio dal training** assumendo che, in fase di inferenza, si utilizzi, **a monte del processamento del segnale tramite il modello**, un sistema di **eliminazione a priori degli intervalli di silenzio troppo lunghi** (esempio tramite utilizzo di un VAD o semplicemente tramite analisi della potenza del segnale audio - soluzione da noi adottata -).
3. Inizialmente, al fine di generare delle **clip della classe negativa**, avevamo scelto di **utilizzare un secondo dataset**, contenente dei file tratti da audiolibri (LibriSpeech ASR corpus: [https://www.openslr.org/12]). Il motivo dietro questa scelta può essere giustificato dal fatto che, generalmente, gli audiolibri sono caratterizzati da una dizione particolarmente accurata, tanto da poter fare l'assunzione che tali segnali audio fossero naturalmente privi di ogni possibile filler word. **Questa soluzione, tuttavia, si è rivelata inefficace in fase di training**, perché abbiamo osservato che l'utilizzo di due dataset con caratteristiche dei dati estremamente differenti inducevano semplicemente il modello a **operare una semplice separazione dei dataset** invece di concentrarsi sulle reali caratteristiche distintive di ogni classe. Questo fenomeno è apparso evidente nel momento in cui sono state analizzate le metriche di valutazione in presenza di trasformazioni per la data augmentation.


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





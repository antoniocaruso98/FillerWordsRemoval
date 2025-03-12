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

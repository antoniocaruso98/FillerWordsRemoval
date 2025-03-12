# Idea di base

Dato un file audio, si vuole identificare la posizione di ogni filler word per poi eventualmente rimuoverla in caso positivo.

Visto che individuare la posizione esatta in termini di timestamp iniziale e finale nella traccia audio della parola è un problema di regressione, quindi complicato da affrontare con le risorse a disposizione,
si è pensato di approssimare la regressione stessa con un task di classificazione su una finestra scorrevole. Per ogni finestra si vuole capire se essa contenga o meno una filler word e classificarla di conseguenza.

I timestamp iniziali e finali della parola coincidono (approssimativamente) con quelli iniziali e finali della sliding window corrente.

Una volta individuate le sliding window positive, queste devono essere tagliate dall'audio originale.

Complessivamente il task è assimilabile a una object detection applicata all'audio.

## Metriche di valutazione

- IoU

> Nota: Per valutare la IoU, usare strumento già esistente per calcolare il ground truth (etichettare il dato originale con la posizione reale delle filler words), oppure usare un dataset già etichettato.

- Accuracy
- Precision
- Recall

(Le classi sono sbilanciate, perchè quella negativa è molto più frequente)

## Dataset

- Training: dataset con classe binaria, a livello di singola word (dato un audio con una sola parola -> Filler o no);
- Test: dataset con frasi e timestamp di inizio e fine per ogni filler word; (da verificare)

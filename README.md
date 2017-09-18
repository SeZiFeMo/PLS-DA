# PLS-DA
## Progetto d'esame di Federico Motta e Serena Ziviani

### Installare Conda
[Conda](https://conda.io/docs/index.html) è un gestore di pacchetti e _environment_ python. Automatizza e semplifica
il processo di installazione e gestione di ambienti python separati sulla
stessa macchina.
In caso di dubbi è possibile consultare la [documentazione ufficiale](https://conda.io/docs/index.html)
1. scaricare __miniconda__ dal [sito](https://conda.io/miniconda.html)
2. scegliere la versione per __python 3.6__ per il sistema operativo in uso
3. eseguire il file scaricato e seguire le istruzioni a schermo

### Installazione dei requisiti tramite conda
I requisiti del programma sono contenuti nel file __environment.yml__.

1. Comando per creare un nuovo environment Conda coi pacchetti richiesti.
`conda env create -f environment.yml`

2. Per accedere all'environment:
`activate MottaZivianiPLSDA`

### Lanciare il programma
1. Entrare nell'ambiente conda: ```activate MottaZivianiPLSDA```
2. Lanciare il comando:
```...```

## FORMATO CSV
Il file di input deve essere conforme allo standard dei file "comma separated
value", avere quindi il punto e virgola come separatore ed essere salvato con
codifica iso8859 (quella standard per l'italiano).
La prima colonna deve contenere il tipo "Category", mentre le seguenti
specificano le variabili.
La prima riga deve contenere le etichette delle variabili.

Category;var1;var2;...;varM
cat1;val11;val12;...;val1M
        ...
catX;valN1;valN2;...;valNM


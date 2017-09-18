# PLS-DA
## Progetto d'esame di Federico Motta e Serena Ziviani

### Installare Conda
[Conda](https://conda.io/docs/index.html) è un gestore di pacchetti e _environment_ python. Automatizza e semplifica il processo di installazione e gestione di ambienti python separati sulla stessa macchina.
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

## CSV FORMAT
The input file must be a standard comma separated value file, with comma
as a separator and saved with iso8859 encoding (standard Italian encoding).
The first column must be the Category type, while the others specify the
variables.
The first row must be the label of the variables

Category;var1;var2;...;varM
cat1;val11;val12;...;val1M
        ...
catX;valN1;valN2;...;valNM


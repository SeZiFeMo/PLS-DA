# PLS-DA
## A Python application to perform PLS Discriminant Analysis

### Install Conda
[Conda](https://conda.io/docs/index.html) is an environment and package manager for Python. It simplifies and automates the installation and management of different python environments on the same host.
Please consult the [official documentation](https://conda.io/docs/index.html) for mode details.
1. download __miniconda__ [here](https://conda.io/miniconda.html)
2. choose the __python 3.6__ version for the operative system in use
3. execute the downloaded file and follow the displayed instructions

### Install requirements with conda
Requirements list is in the file __environment.yml__.

1. Create a new environment in conda with the specified packages:
`conda env create -f environment.yml`

2. Then activate it:
`activate MottaZivianiPLSDA`

### Execute the application
1. Activate conda environment: ```activate MottaZivianiPLSDA```
2. Run the following command:
```python pls-da.py```

## CSV FORMAT
The input files have to be compliant with the _comma separated value_ standard, thus the separator used is the semicolon ';' and the encoding of the file is ISO8859 (the standard one in Italy).
The first row have to have the variable labels.
The first column have to be the "Category" type, while the others the variables values.

Category;var1;var2;...;varM

cat1;val11;val12;...;val1M

...

catX;valN1;valN2;...;valNM

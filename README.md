# Modelos I - Proyecto
| Nombre | Cédula | Programa |
|----------|----------|----------|
| Andrea Sánchez Castrillón| 1001420939 | Ingeniería de sistemas   |
| Alejandro Vargas Ocampo   | 1088298091   | Ingeniería de sistemas   |
### Fuente del challenge:
Predicting molecular properties. 
https://www.kaggle.com/competitions/champs-scalar-coupling/code
## Fase-1
The notebook does not require any additional elements, just ALL the lines of code must be executed in the order in which they are.


## Fase-2

1. Download train.csv from here: https://drive.google.com/uc?id=1cFjgpNoWLIFtvfBKqsxeZ_vWNeuelhwh this one doesn't appear because is too large.

2. To run the application correctly, you need to have Docker installed and build a Docker image from the Dockerfile file.

Once the image is built, it is only necessary to run it
### 01_generate_data_and_model
In this file you can see how the process that was delivered in phase-1 is simplified, and how we use the data to train a mode. This logic is what is used later to create train.py

3. Run the notebook `01 - generate data and model` to generate sample train and test data, and see how models are stored and retrieved
### 02_run_scripts
In this file we can see how train.py and predict.py are used from a console.

4. Run the notebook `02 - run scripts` to see how scripts are run

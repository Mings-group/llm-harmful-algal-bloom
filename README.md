# LLM-Assisted Development of Deep Learning Models for Forecasting Diarrhetic Shellfish Poisoning (DSP) Events

## Overview

This repository contains the code, datasets, and supporting materials used in the study:

**Using Large Language Models to Automate Harmful Algal Bloom Prediction**

The objective of this work is to forecast concentrations of diarrhetic shellfish poisoning (DSP) toxins using machine learning and deep learning models developed with assistance from large language models (LLMs). The repository includes preprocessing workflows, model implementations, hyperparameter experiments, evaluation scripts, and supporting datasets.

The repository also contains archived conversations used during model development to document how LLM assistance was incorporated into the research workflow.

---

## Repository Structure

```text
llm-harmful-algal-bloom/

├── preprocessing_scripts/ # Data preparation and feature engineering
│   ├── create_master_data_file.py
│   ├── combine_phyto_toxin.py
│   ├── prepare_SST_data.py
│   ├── prepare_meteo_data.py
│   ├── preprocess_10yr.py
│   └── preprocess_classification_10yr.py
│
├── CNN_model/ # CNN forecasting model implementation
│   ├── model.py
│   └── plot_RIAV1.py
│
├── MLP_model/ # MLP forecasting model implementation
│   └── MLP_model.py
│
├── classification_model/ # DSP event classification models
│   ├── model.py
│   ├── model_10yr.py
│   └── model.ipynb
│
├── experiments_LSTM_model/ # LSTM forecasting and sensitivity analyses
│   ├── learning_rate_experiment.py
│   ├── LSTM_windowing_experiment.py
│   ├── LSTM_regularization_experiment.py
│   ├── LSTM_hidden_layer_experiment.py
│   ├── LSTM_neuron_experiment.py
│   ├── LSTM_batch_experiment.py
│   └── LSTM_MSE_loss_optimization.py
│
├── data/ # Raw and processed datasets
│   ├── all_sites/
│   ├── phytoplankton_data/
│   ├── meteorological_data/
│   ├── SST_data/
│   └── MASTER_DATA_FILE.csv
│
└── ChatGPT Conversations/ # Archived LLM-assisted development records
```

---

## Data Sources

This study uses environmental, phytoplankton, and shellfish toxin datasets collected from multiple shellfish production areas.

The repository includes data associated with the following production areas:

* RIAV1
* L5b
* ETJ1
* LAL
* L7c1
* LAG
* POR2
* FAR1

The datasets used in this study were obtained from the following sources:

### Portuguese Institute for the Sea and Atmosphere (IPMA)

The following datasets were provided by IPMA:

- Diarrhetic Shellfish Poisoning (DSP) toxin measurements
- Phytoplankton observations
- Meteorological observations

These data were collected as part of routine environmental and shellfish monitoring programs conducted by IPMA.

### Copernicus Marine Service

The following remotely sensed and modeled oceanographic variables were obtained from the Copernicus Marine Service:

- Sea Surface Temperature (SST)
- Chlorophyll-a concentration

Copernicus Marine products are publicly available through the Copernicus Marine Data Store.

The modeling framework incorporates:

* DSP toxin measurements
* Phytoplankton observations
* Sea surface temperature (SST)
* Meteorological variables
* Chlorophyll-related variables

## Data Availability

The processed datasets required to reproduce the experiments reported in the manuscript are included in this repository.

Original environmental observations were obtained from the Portuguese Institute for the Sea and Atmosphere (IPMA) and Copernicus Marine Service. Users should acknowledge and cite the Portuguese Institute for the Sea and Atmosphere (IPMA) and Copernicus Marine Service when reusing these datasets.

---

## Software Requirements

The code was developed using Python.

Install required packages using:

```bash
pip install -r requirements.txt
```

Major dependencies include:

* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Statsmodels
* NetCDF4
* Xarray

---

## Reproducing the Workflow

### Step 1: Prepare Environmental Data

Run the preprocessing scripts to combine and standardize the environmental datasets:

```bash
python preprocessing_scripts/combine_dsp_toxin_file.py
python preprocessing_scripts/combine_phyto_toxin.py
python preprocessing_scripts/prepare_SST_data.py
python preprocessing_scripts/prepare_meteo_data.py
```

### Step 2: Create the Master Dataset

Generate the merged dataset used for model development:

```bash
python preprocessing_scripts/create_master_data_file.py
```

### Step 3: Generate Model Inputs

Create the forecasting datasets used by the machine learning models:

```bash
python preprocessing_scripts/preprocess_10yr.py
```

For classification experiments:

```bash
python preprocessing_scripts/preprocess_classification_10yr.py
```

### Step 4: Train Models

#### Multilayer Perceptron (MLP)

```bash
python MLP_model/MLP_model.py
```

#### Convolutional Neural Network (CNN)

```bash
python CNN_model/model.py
```

### Long Short Term Memory (LSTM)
```bash
python experiments_LSTM_model/LSTM_MSE_loss_optimization.py
```
#### Classification Model

run classification_model/model.ipynb
```bash
python classification_model/model_10yr.py
```

### Step 5: Hyperparameter Experiments

LSTM sensitivity experiments can be performed using the scripts in:

```text
experiments_LSTM_model/
```

These scripts investigate:

* Learning rate
* Batch size
* Window length
* Number of hidden layers
* Number of neurons
* Regularization strategies
* Loss functions

### Step 6: Evaluation and Visualization

Example visualization scripts are provided, including:

```bash
python CNN_model/plot_RIAV1.py
python experiments_LSTM_model/plot_RIAV1_MSE.py
```

These scripts generate figures and performance summaries used in the manuscript.

---

## Repository Outputs

Running the workflow produces:

* Preprocessed datasets
* Trained model files
* Forecast predictions
* Classification outputs
* Evaluation metrics
* Publication figures

---

## LLM-Assisted Development

This repository contains archived conversations documenting the use of large language models during the development of preprocessing pipelines, model architectures, hyperparameter experiments, and evaluation workflows.

These materials are included to support transparency and reproducibility of the LLM-assisted research process.

---


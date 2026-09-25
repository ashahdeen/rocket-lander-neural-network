# rocket-lander-neural-network

CE889 Artificial Neural Networks — Asha Deen (2501218), Autumn 2025

---

## Overview

This project covers two distinct neural network tasks: a from-scratch backpropagation MLP built to autonomously land a rocket, and a deep learning model trained to forecast retail sales across 1,115 stores. Both are implemented end-to-end — from data collection and preprocessing through to training, evaluation, and deployment.

---

## Part 1 — Backpropagation MLP (Lander Game)

### Problem Statement
A rocket lander game generates a random landing zone and unsafe terrain on each run. The objective is to land the ship safely without touching unsafe terrain. Data collected from manual gameplay is used to train a neural network that can then replicate the task autonomously.

### Objectives
1. Build a feed-forward backpropagation MLP from scratch using only Python — no ML libraries.
2. Train the network offline on manually collected gameplay data (~100 epochs).
3. Evaluate performance by running the trained weights in feed-forward mode and computing RMSE.

### Approach
- **Data collection** — played the lander game manually; inputs (X, Y offset to target) and outputs (thruster, turning) recorded to `ce889_dataCollection.csv`
- **Preprocessing** — deduplication, min-max normalisation, 70/15/15 train/val/test split
- **Architecture** — 2 → 7 → 2 (sigmoid activation throughout)
- **Training** — gradient descent with momentum (lr=0.5, momentum=0.5, 100 epochs, MSE loss)
- **Inference** — trained weights hardcoded into `NeuralNetHolder.py` and loaded at game runtime

### Files

| File | Description |
|------|-------------|
| [`assignment-code/NeuralNetHolder.py`](assignment-code/NeuralNetHolder.py) | Trained MLP weights + `predict()` — loaded by the game |
| [`lander_online/train_lander.py`](lander_online/train_lander.py) | Full training pipeline: load, split, train, save weights |
| [`lander_online/ce889_dataCollection.csv`](lander_online/ce889_dataCollection.csv) | Raw gameplay data collected manually |
| [`lander_online/lander_params.txt`](lander_online/lander_params.txt) | Saved trained weights and normalisation parameters |
| [`lander_online/train_split.csv`](lander_online/train_split.csv) | Training split (70%) |
| [`lander_online/val_split.csv`](lander_online/val_split.csv) | Validation split (15%) |
| [`lander_online/test_split.csv`](lander_online/test_split.csv) | Test split (15%) |
| [`assignment-code/Main.py`](assignment-code/Main.py) | Game entry point |

### Tools
Pure Python 3 — no external ML libraries. `pygame` for the game GUI.

---

## Part 2 — Deep Learning (Rossmann Store Sales)

### Problem Statement
Rossmann operates over 1,100 drug stores across Germany. Store managers are required to forecast daily sales up to six weeks in advance, but individual forecasts vary significantly due to differences in promotions, competition, school holidays, and local conditions. The goal is to predict sales accurately across all stores using historical data.

### Objectives
1. Develop a deep neural network trained on historical Rossmann store data.
2. Explore and compare multiple architectures (DNN, LSTM, GRU) to identify the best performer.
3. Generate a final submission for the [Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales).

### Approach
- **Preprocessing** — merged store metadata, handled missing values, engineered date features (day of week, month, promo flags), scaled numerical features
- **Models explored** — baseline DNN, LSTM, GRU, and tuned variants with early stopping and log-scale loss
- **Evaluation** — Root Mean Square Percentage Error (RMSPE), consistent with Kaggle's scoring metric
- **Best model** — DNN with log-transformed target and early stopping (`best_dnn.weights.h5`)

### Files

| File | Description |
|------|-------------|
| [`rossmann-store-sales/rossman_prep.ipynb`](rossmann-store-sales/rossman_prep.ipynb) | Data cleaning and feature engineering |
| [`rossmann-store-sales/train_ross.ipynb`](rossmann-store-sales/train_ross.ipynb) | Model training — DNN, LSTM, GRU |
| [`rossmann-store-sales/final_models.ipynb`](rossmann-store-sales/final_models.ipynb) | Model comparison and final selection |
| [`rossmann-store-sales/submission_2.ipynb`](rossmann-store-sales/submission_2.ipynb) | Kaggle submission generation |
| [`rossmann-store-sales/submission_final_dnn.csv`](rossmann-store-sales/submission_final_dnn.csv) | Final Kaggle submission |
| [`rossmann-store-sales/best_dnn.weights.h5`](rossmann-store-sales/best_dnn.weights.h5) | Best DNN weights |
| [`rossmann-store-sales/best_lstm_model.h5`](rossmann-store-sales/best_lstm_model.h5) | Best LSTM model |
| [`rossmann-store-sales/store.csv`](rossmann-store-sales/store.csv) | Store metadata |
| `rossmann-store-sales/train.csv` | Training data — excluded (36MB). Download from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) |

### Tools
Python 3.11, TensorFlow / Keras, pandas, NumPy, scikit-learn, Jupyter.

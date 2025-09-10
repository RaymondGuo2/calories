# Neural Network Regression for Predicting Calorie Expenditure

This repository contains code for training and evaluating a neural network model on the [Kaggle competition dataset](https://www.kaggle.com/competitions/playground-series-s5e5/overview).  

The task is framed as a regression problem, and performance is measured using **Root Mean Squared Logarithmic Error (RMSLE)**.

## Project Structure
- `data.py` – Data loading and preprocessing utilities  
- `train.py` – Training loop and evaluation logic  
- `model.py` – Neural network architecture definition  
- `loss.py` – Custom loss functions and metrics  
- `genetic_algorithm.py` – Hyperparameter search using a genetic algorithm  
- `generate_submission.py` – Script for generating competition-ready submission files  
- `data_vis.ipynb` – Exploratory data analysis and visualization notebook  
- `main.py` – Entry point to run training and evaluation  

## Approach
- A feedforward neural network was implemented to model the target variable.  
- Hyperparameters were tuned with a genetic algorithm to improve performance.  
- The model was trained on preprocessed features and evaluated using **RMSLE**.  
- Final predictions were exported in the required submission format.  
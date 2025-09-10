# The purpose of this code is to execute the main function of the code
from data import load_data
import torch.nn as nn
import torch.optim as optim
from model import Net
import torch
from train import train, evaluate

def main(train_data_path):
    # Step 1: Pre-process Data
    X_train, y_train, X_val, y_val = load_data(train_data_path, data='train', val_size=0.25)
    
    # Train the model
    model = train(X_train, y_train, num_epochs=1000)
    evaluate(model, X_val, y_val)


if __name__ == '__main__':
    train_data = '/Users/raymondguo/Desktop/datasci/calories/playground-series-s5e5/train.csv'
    main(train_data)
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(dataset_path, data='train', val_size=0.25):
    df = pd.read_csv(dataset_path)
    df = df.drop(['id'], axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) 

    # Since these values were quite skewed in the distribution, they will be log-transformed
    df['Age'] = np.log1p(df['Age'])
    
    if data == 'train':
        # Separate predictors and outcome
        X = df.drop(columns=['Calories'])
        y = df['Calories'].values           

        # Separate items to standardise and those not to
        sex = X[['Sex']].values
        numeric_features = X.drop(columns = ['Sex']).values

        X_train_num, X_val_num, sex_train, sex_val, y_train, y_val = train_test_split(
            numeric_features, sex, y, test_size=val_size)

        # Standardize numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_num)
        X_val_scaled = scaler.transform(X_val_num)

        # Combine scaled features + raw Sex
        X_train = np.concatenate([X_train_scaled, sex_train], axis=1)
        X_val = np.concatenate([X_val_scaled, sex_val], axis=1)

        # Convert data to tensors for training
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # print(f"Training data loaded ✅. There are {len(X_train)} training samples and {len(X_val)} validation samples")
        return X_train, y_train, X_val, y_val
    else:
        # Separate features for scaling
        X = df
        sex = X[['Sex']].values
        numeric_features = X.drop(columns = ['Sex']).values

        # Standardize numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_features)

        # Combine scaled features + raw Sex
        X_test = np.concatenate([X_scaled, sex], axis=1)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        print(f"Test data loaded ✅. There are {len(X_test)} test samples")
        return X_test, None

    
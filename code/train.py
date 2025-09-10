import torch.nn as nn
import torch.optim as optim
from model import Net
import torch
import matplotlib.pyplot as plt
from loss import RMSLELoss

def train(X_train, y_train, num_epochs=100, patience=5, min_delta=1e-4):
    loss_fn = RMSLELoss()
    model = Net(hidden_layer=256, dropout=0.18)
    optimiser = optim.Adam(model.parameters(), lr=0.00958, weight_decay=0.001)
    losses = []

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimiser.step()
        current_loss = loss.item()
        losses.append(loss.item())

        print(f"Epoch {epoch + 1} returned a loss of: {current_loss:.6f}")

        if abs(best_loss - current_loss) < min_delta:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            best_loss = current_loss
            epochs_no_improve = 0
    print("Training completed")
    torch.save(model.state_dict(), 'model_v5.pth')

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

def evaluate(model, X_eval, y_eval):
    model.eval()
    loss_fn = RMSLELoss()
    with torch.no_grad():
        outputs = model(X_eval)
        loss = loss_fn(outputs, y_eval)
        print("RMSLE on evaluation data:", loss.item())

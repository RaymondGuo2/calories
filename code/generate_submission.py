from model import Net
import torch
from data import load_data
import csv

model = Net(hidden_layer=256, dropout=0.18)
model.load_state_dict(torch.load('model_v5.pth'))
model.eval()

test_data = '/Users/raymondguo/Desktop/datasci/calories/playground-series-s5e5/test.csv'
X_test, _ = load_data(test_data, data='test')

with torch.no_grad():
    predictions = model(X_test).squeeze()

start_id = 750000
predictions = predictions.numpy()

with open('predictions_3.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'Calories'])

    for i, pred in enumerate(predictions):
        writer.writerow([start_id + i, f"{pred:.3f}"])
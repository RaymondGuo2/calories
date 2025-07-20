import torch
import torch.optim as optim
import random
from model import Net
from loss import RMSLELoss
from data import load_data
import matplotlib.pyplot as plt

def fitness(chromosome, data_path):

    X_train, y_train, X_val, y_val = load_data(data_path, data='train', val_size=0.25)

    lr, hidden_units, dropout = chromosome
    model = Net(hidden_layer=hidden_units, dropout=dropout)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = RMSLELoss()

    model.train()
    for epoch in range(3):
        optimiser.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimiser.step()
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
    
    return -val_loss.item()

def generate_chromosome():
    return [random.uniform(0.0001, 0.01), random.choice([64, 128, 256]), random.uniform(0.0, 0.5)]

def crossover(p1, p2):
    point = random.randint(1, 2)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(c, rate=0.1):
    if random.random() < rate:
        c[0] = random.uniform(0.0001, 0.01)
    if random.random() < rate:
        c[1] = random.choice([64, 128, 256])
    if random.random() < rate:
        c[2] = random.uniform(0.0, 0.5)
    return c

if __name__ == '__main__':
    generations, pop_size, retain = 5, 10, 5
    population = [generate_chromosome() for _ in range(pop_size)]
    history = []

    train_path = '/Users/raymondguo/Desktop/datasci/calories/playground-series-s5e5/train.csv'
    for g in range(generations):
        scores = [fitness(c, train_path) for c in population]
        best = max(scores)
        history.append(best)
        print(f"\nGeneration {g+1}/{generations}")
        print(f" Best fitness (negative val loss): {best:.4f}")
        best_chrom = population[scores.index(best)]
        print(f" Best chromosome: lr={best_chrom[0]:.5f}, hidden={int(best_chrom[1])}, dropout={best_chrom[2]:.2f}")
        sorted_pop = [x for _, x in sorted(zip(scores, population), reverse=True)]
        parents = sorted_pop[:retain]

        next_gen = []
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(c2))
        population = next_gen

    # Results
    final_scores = [fitness(c, train_path) for c in population]
    best_idx = final_scores.index(max(final_scores))
    best = population[best_idx]

    print("\nBest Hyperparameters:")
    print(f" Learning rate: {best[0]:.5f}")
    print(f" Hidden units: {int(best[1])}")
    print(f" Dropout rate: {best[2]:.2f}")

    # Plot accuracy over generations
    plt.plot(history, marker='o')
    plt.title("Best Validation Accuracy Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
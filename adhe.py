import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_limits

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
df = pd.read_csv('/Users/david_h/Downloads/medication_adherence.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the data into features and target variable
X = df.drop('adherence', axis=1).values
y = df['adherence'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Increased neurons
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()

# Define Optimization Algorithms

def randomized_hill_climbing(model, X_train, y_train, criterion, max_iters=300):
    best_weights = {name: param.clone() for name, param in model.named_parameters()}
    best_fitness = float('-inf')

    for _ in range(max_iters):
        new_weights = {name: param + torch.randn_like(param) * 0.05 for name, param in model.named_parameters()}
        model.load_state_dict(new_weights)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        fitness = -loss.item()  # Using negative loss as fitness

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = new_weights

    return best_weights, best_fitness

def simulated_annealing(model, X_train, y_train, criterion, initial_temp=1500, cooling_rate=0.85, max_iters=300):
    best_weights = {name: param.clone() for name, param in model.named_parameters()}
    current_weights = best_weights
    best_fitness = float('-inf')
    current_fitness = best_fitness
    temp = initial_temp

    for _ in range(max_iters):
        new_weights = {name: param + torch.randn_like(param) * 0.05 for name, param in model.named_parameters()}
        model.load_state_dict(new_weights)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        fitness = -loss.item()  # Using negative loss as fitness

        if fitness > current_fitness or np.exp((fitness - current_fitness) / temp) > np.random.rand():
            current_fitness = fitness
            current_weights = new_weights

            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = new_weights

        temp *= cooling_rate

    return best_weights, best_fitness

def genetic_algorithm(model, X_train, y_train, criterion, solution_length, pop_size=100, generations=300, mutation_rate=0.2):
    def crossover(parent1, parent2):
        point = np.random.randint(1, len(parent1) - 1)
        return torch.cat((parent1[:point], parent2[point:]))

    def mutate(solution):
        idx = np.random.randint(0, len(solution))
        solution[idx] += torch.normal(mean=0, std=0.1, size=(1,)).item()  # .item() converts tensor to scalar
        return solution

    def fitness(weights):
        new_weights = {name: torch.tensor(w, dtype=torch.float32).view(param.shape) for (name, param), w in zip(model.named_parameters(), np.split(weights, np.cumsum([param.numel() for param in model.parameters()])[:-1]))}
        model.load_state_dict(new_weights)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        return -loss.item()  # Using negative loss as fitness

    population = [torch.tensor(np.random.randn(solution_length), dtype=torch.float32) for _ in range(pop_size)]

    for _ in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        next_gen = population[:pop_size // 2]

        while len(next_gen) < pop_size:
            parents = np.random.choice(len(next_gen), 2, replace=False)
            parent1, parent2 = next_gen[parents[0]], next_gen[parents[1]]
            offspring = crossover(parent1, parent2)
            if np.random.rand() < mutation_rate:
                offspring = mutate(offspring)
            next_gen.append(offspring)

        population = next_gen

    best_solution = max(population, key=fitness)
    return best_solution, fitness(best_solution)

with threadpool_limits(limits=1):
    # Initial random weights
    initial_weights = torch.cat([param.view(-1) for param in model.parameters()]).detach().numpy()

    # Randomized Hill Climbing
    best_weights_rhc, best_fitness_rhc = randomized_hill_climbing(model, X_train, y_train, criterion)
    model.load_state_dict(best_weights_rhc)
    y_pred = model(X_test).argmax(dim=1)
    accuracy_rhc = accuracy_score(y_test, y_pred)
    print(f"RHC Best Fitness: {best_fitness_rhc}, Test Accuracy: {accuracy_rhc}")

    # Simulated Annealing
    best_weights_sa, best_fitness_sa = simulated_annealing(model, X_train, y_train, criterion)
    model.load_state_dict(best_weights_sa)
    y_pred = model(X_test).argmax(dim=1)
    accuracy_sa = accuracy_score(y_test, y_pred)
    print(f"SA Best Fitness: {best_fitness_sa}, Test Accuracy: {accuracy_sa}")

    # Genetic Algorithm
    best_weights_ga, best_fitness_ga = genetic_algorithm(model, X_train, y_train, criterion, solution_length=len(initial_weights))
    new_weights_ga = {name: torch.tensor(w, dtype=torch.float32).view(param.shape) for (name, param), w in zip(model.named_parameters(), np.split(best_weights_ga.numpy(), np.cumsum([param.numel() for param in model.parameters()])[:-1]))}
    model.load_state_dict(new_weights_ga)
    y_pred = model(X_test).argmax(dim=1)
    accuracy_ga = accuracy_score(y_test, y_pred)
    print(f"GA Best Fitness: {best_fitness_ga}, Test Accuracy: {accuracy_ga}")

    # Plot Learning Curve

    # Define a wrapper class to use PyTorch model with scikit-learn functions
    class NeuralNetWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            optimizer = torch.optim.Adam(self.model.parameters())
            for epoch in range(100):  # A simple training loop
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            self.classes_ = torch.unique(y).numpy()  # Add classes_ attribute
            return self

        def predict(self, X):
            X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X)
            _, y_pred = torch.max(outputs, 1)
            return y_pred.numpy()

    wrapped_model = NeuralNetWrapper(model)

    # Learning curve for Neural Network
    train_sizes, train_scores, valid_scores = learning_curve(
        wrapped_model, X_train.numpy(), y_train.numpy(), cv=5, scoring='accuracy', n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, valid_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for Neural Network')
    plt.legend()

    plt.tight_layout()
    plt.show()


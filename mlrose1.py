import mlrose_hiive as mlrose
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import time
import random

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load and preprocess the dataset
df = pd.read_csv('/Users/david_h/Downloads/medication_adherence.csv')
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('adherence', axis=1).values
y = df['adherence'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
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

# Define the fitness function
def fitness_function(weights):
    weight_idx = 0
    for name, param in model.named_parameters():
        param_shape = param.shape
        param_size = param.numel()
        param_data = weights[weight_idx:weight_idx + param_size].reshape(param_shape)
        param.data = torch.tensor(param_data, dtype=torch.float32)
        weight_idx += param_size
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    return -loss.item()

# Set up the optimization problem
initial_weights = torch.cat([param.view(-1) for param in model.parameters()]).detach().numpy()
fitness = mlrose.CustomFitness(fitness_function)
problem = mlrose.ContinuousOpt(length=len(initial_weights), fitness_fn=fitness, maximize=True)

# Define optimization function using mlrose
def optimize_with_mlrose(problem, algorithm, max_iters=300, initial_temp=1500, cooling_rate=0.85, pop_size=100, mutation_rate=0.2):
    if algorithm == 'rhc':
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_iters=max_iters, curve=True, random_state=42)
    elif algorithm == 'sa':
        schedule = mlrose.ExpDecay(init_temp=initial_temp, exp_const=cooling_rate)
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule, max_iters=max_iters, curve=True, random_state=42)
    elif algorithm == 'ga':
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mutation_rate, max_iters=max_iters, curve=True, random_state=42)
    else:
        raise ValueError("Algorithm must be 'rhc', 'sa', or 'ga'")
    return best_state, best_fitness, fitness_curve

# Run the optimization algorithms and measure time
results = {}

start_time = time.time()
best_state_rhc, best_fitness_rhc, fitness_rhc = optimize_with_mlrose(problem, 'rhc')
rhc_time = time.time() - start_time
results['RHC'] = (best_fitness_rhc, fitness_rhc, rhc_time)
print(f"RHC Best Fitness: {best_fitness_rhc}, Time: {rhc_time}")

best_weights_rhc = {name: torch.tensor(w, dtype=torch.float32).view(param.shape)
                    for (name, param), w in zip(model.named_parameters(), np.split(best_state_rhc, np.cumsum([param.numel() for param in model.parameters()])[:-1]))}
model.load_state_dict(best_weights_rhc)
y_pred = model(X_test).argmax(dim=1)
accuracy_rhc = accuracy_score(y_test, y_pred)
print(f"RHC Test Accuracy: {accuracy_rhc}")

start_time = time.time()
best_state_sa, best_fitness_sa, fitness_sa = optimize_with_mlrose(problem, 'sa')
sa_time = time.time() - start_time
results['SA'] = (best_fitness_sa, fitness_sa, sa_time)
print(f"SA Best Fitness: {best_fitness_sa}, Time: {sa_time}")

best_weights_sa = {name: torch.tensor(w, dtype=torch.float32).view(param.shape)
                   for (name, param), w in zip(model.named_parameters(), np.split(best_state_sa, np.cumsum([param.numel() for param in model.parameters()])[:-1]))}
model.load_state_dict(best_weights_sa)
y_pred = model(X_test).argmax(dim=1)
accuracy_sa = accuracy_score(y_test, y_pred)
print(f"SA Test Accuracy: {accuracy_sa}")

start_time = time.time()
best_state_ga, best_fitness_ga, fitness_ga = optimize_with_mlrose(problem, 'ga')
ga_time = time.time() - start_time
results['GA'] = (best_fitness_ga, fitness_ga, ga_time)
print(f"GA Best Fitness: {best_fitness_ga}, Time: {ga_time}")

best_weights_ga = {name: torch.tensor(w, dtype=torch.float32).view(param.shape)
                   for (name, param), w in zip(model.named_parameters(), np.split(best_state_ga, np.cumsum([param.numel() for param in model.parameters()])[:-1]))}
model.load_state_dict(best_weights_ga)
y_pred = model(X_test).argmax(dim=1)
accuracy_ga = accuracy_score(y_test, y_pred)
print(f"GA Test Accuracy: {accuracy_ga}")

# Plot Fitness / Iteration
plt.figure(figsize=(12, 6))
for algo, (best_fitness, fitness_curve, time_taken) in results.items():
    plt.plot(fitness_curve[:, 1], label=f'{algo} (Time: {time_taken:.2f}s)')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Fitness Over Iterations')
plt.legend()
plt.show()

# Function Evaluations and Wall Clock Time Comparison
algorithms = ['RHC', 'SA', 'GA']
fitness_values = [results[algo][0] for algo in algorithms]
times = [results[algo][2] for algo in algorithms]

plt.figure(figsize=(12, 6))
plt.bar(algorithms, fitness_values, color='b', alpha=0.6, label='Best Fitness')
plt.xlabel('Algorithm')
plt.ylabel('Fitness')
plt.title('Best Fitness Comparison')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(algorithms, times, color='r', alpha=0.6, label='Time Taken (s)')
plt.xlabel('Algorithm')
plt.ylabel('Time (s)')
plt.title('Time Taken for Optimization')
plt.legend()
plt.show()

# Test Different Problem Sizes
problem_sizes = [len(initial_weights) // 2, len(initial_weights), len(initial_weights) * 2]
size_results = {}

for size in problem_sizes:
    size_results[size] = {}
    problem = mlrose.ContinuousOpt(length=size, fitness_fn=fitness, maximize=True)
    for algo in ['rhc', 'sa', 'ga']:
        start_time = time.time()
        best_state, best_fitness, fitness_curve = optimize_with_mlrose(problem, algo, max_iters=300)
        size_results[size][algo] = best_fitness
        print(f"Problem Size: {size}, Algorithm: {algo}, Best Fitness: {best_fitness}")

# Plot Fitness / Problem Size
plt.figure(figsize=(12, 6))
for algo in ['rhc', 'sa', 'ga']:
    fitness_values = [size_results[size][algo] for size in problem_sizes]
    plt.plot(problem_sizes, fitness_values, marker='o', label=algo)
plt.xlabel('Problem Size')
plt.ylabel('Best Fitness')
plt.title('Best Fitness vs Problem Size')
plt.legend()
plt.show()




import random
import math

# Sparse Peaks fitness function
def sparse_peaks_fitness(bit_string):
    fitness = 0
    current_run = 0
    
    for bit in bit_string:
        if bit == 1:
            current_run += 1
        else:
            if current_run == 1:
                fitness += 10  # Reward isolated peaks
            elif current_run > 1:
                fitness -= 5  # Penalize runs longer than 1
            current_run = 0

    if current_run == 1:
        fitness += 10
    elif current_run > 1:
        fitness -= 5
    
    return fitness

# Modular Graph Coloring fitness function
def modular_graph_coloring_fitness(graph, coloring):
    fitness = 0
    
    for module in graph['modules']:
        for (u, v) in module['edges']:
            if coloring[u] != coloring[v]:
                fitness += 2  # Higher penalty for intra-module edges

    for (u, v) in graph['inter_module_edges']:
        if coloring[u] != coloring[v]:
            fitness += 1  # Lower penalty for inter-module edges

    return fitness

# Example modular graph structure
modular_graph = {
    'modules': [
        {'edges': [(0, 1), (1, 2), (2, 0)]},
        {'edges': [(3, 4), (4, 5), (5, 3)]}
    ],
    'inter_module_edges': [(2, 3)]
}

# Randomized Hill Climbing
def hill_climbing(fitness_fn, initial_state, max_iterations=1000):
    current_state = initial_state
    current_fitness = fitness_fn(current_state)

    for _ in range(max_iterations):
        candidate_state = mutate_bitstring(current_state[:])
        candidate_fitness = fitness_fn(candidate_state)

        if candidate_fitness > current_fitness:
            current_state = candidate_state
            current_fitness = candidate_fitness

    return current_state, current_fitness

# Simulated Annealing
def annealing(fitness_fn, initial_state, initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    current_state = initial_state
    current_fitness = fitness_fn(current_state)
    temperature = initial_temp

    for _ in range(max_iterations):
        candidate_state = mutate_bitstring(current_state[:])
        candidate_fitness = fitness_fn(candidate_state)

        if candidate_fitness > current_fitness or math.exp((candidate_fitness - current_fitness) / temperature) > random.random():
            current_state = candidate_state
            current_fitness = candidate_fitness

        temperature *= cooling_rate

    return current_state, current_fitness

# Genetic Algorithm
def genetic_algo(fitness_fn, population_size=100, generations=1000, mutation_rate=0.01, solution_length=30):
    population = [random_bitstring(solution_length) for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=fitness_fn, reverse=True)
        next_generation = population[:population_size // 2]

        while len(next_generation) < population_size:
            parent1 = random.choice(next_generation)
            parent2 = random.choice(next_generation)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate_bitstring(child)
            next_generation.append(child)

        population = next_generation

    best_solution = max(population, key=fitness_fn)
    return best_solution, fitness_fn(best_solution)

# Helper Functions
def mutate_bitstring(bitstring):
    index = random.randint(0, len(bitstring) - 1)
    bitstring[index] = 1 - bitstring[index]
    return bitstring

def random_bitstring(length):
    return [random.randint(0, 1) for _ in range(length)]

def random_coloring(num_vertices, num_colors):
    return [random.randint(0, num_colors - 1) for _ in range(num_vertices)]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Applying the Algorithms to Sparse Peaks Problem
print("Sparse Peaks Problem:")
initial_state_sp = random_bitstring(30)
print(f"Initial state: {initial_state_sp}")

best_hc_sp, fitness_hc_sp = hill_climbing(sparse_peaks_fitness, initial_state_sp)
print(f"RHC Best Solution: {best_hc_sp}, Fitness: {fitness_hc_sp}")

best_sa_sp, fitness_sa_sp = annealing(sparse_peaks_fitness, initial_state_sp)
print(f"SA Best Solution: {best_sa_sp}, Fitness: {fitness_sa_sp}")

best_ga_sp, fitness_ga_sp = genetic_algo(sparse_peaks_fitness, population_size=100, generations=1000, solution_length=30)
print(f"GA Best Solution: {best_ga_sp}, Fitness: {fitness_ga_sp}")

# Applying the Algorithms to Modular Graph Coloring Problem
print("\nModular Graph Coloring Problem:")
initial_coloring_mg = random_coloring(len(modular_graph['modules'][0]['edges']) * 2, 3)
print(f"Initial coloring: {initial_coloring_mg}")

best_hc_mg, fitness_hc_mg = hill_climbing(lambda x: modular_graph_coloring_fitness(modular_graph, x), initial_coloring_mg)
print(f"RHC Best Coloring: {best_hc_mg}, Fitness: {fitness_hc_mg}")

best_sa_mg, fitness_sa_mg = annealing(lambda x: modular_graph_coloring_fitness(modular_graph, x), initial_coloring_mg)
print(f"SA Best Coloring: {best_sa_mg}, Fitness: {fitness_sa_mg}")

best_ga_mg, fitness_ga_mg = genetic_algo(lambda x: modular_graph_coloring_fitness(modular_graph, x), population_size=100, generations=1000, solution_length=len(initial_coloring_mg))
print(f"GA Best Coloring: {best_ga_mg}, Fitness: {fitness_ga_mg}")

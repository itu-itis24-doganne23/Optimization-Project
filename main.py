import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Keep for potential plotting, though not used in main script

# --- User-defined Parameters for the Objective Function ---
alpha_prime = 1.0  # Weight for cost component
gamma_prime = 1.0  # Weight for air quality benefit
beta_prime = 1.0   # Weight for transportation benefit
epsilon = 1e-6     # Small constant to avoid division by zero
delta = 0.01       # Minimum green space per capita (e.g., 10 sqm/person)

# --- Data Loading and Preparation ---
try:
    df_districts = pd.read_csv('result/birlesik_ilce_verisi.csv')
    # !!! IMPORTANT: Adjust these column names to match your CSV file EXACTLY !!!
    P_values = df_districts['Nufus'].values
    C_values = df_districts['AraziMaliyeti'].values
    AQ_values = df_districts['HavaKalitesiPuani'].values
    T_values = df_districts['UlasimPuani'].values
    GA_values = df_districts['MevcutYesilAlan'].values
    A_max_values = df_districts['MaxYeniYesilAlan'].values # Max *new* green space
    district_names = df_districts['IlceAdi'].values if 'IlceAdi' in df_districts.columns else [f"District {i+1}" for i in range(len(P_values))]

except FileNotFoundError:
    print("Warning: 'birlesik_ilce_verisi.csv' not found. Using placeholder data.")
    N_mock = 5 # Number of districts for placeholder
    P_values = np.random.randint(50000, 200000, N_mock)
    C_values = np.random.uniform(1000, 5000, N_mock)
    AQ_values = np.random.uniform(20, 100, N_mock) # Higher is worse
    T_values = np.random.uniform(1, 10, N_mock)   # Higher is better
    GA_values = np.random.uniform(10000, 50000, N_mock)
    A_max_values = np.random.uniform(5000, 20000, N_mock)
    district_names = [f"Mock District {i+1}" for i in range(N_mock)]
except KeyError as e:
    print(f"Error: Column {e} not found in CSV. Please check column names. Using placeholder data.")
    N_mock = 5
    P_values = np.random.randint(50000, 200000, N_mock)
    C_values = np.random.uniform(1000, 5000, N_mock)
    AQ_values = np.random.uniform(20, 100, N_mock)
    T_values = np.random.uniform(1, 10, N_mock)
    GA_values = np.random.uniform(10000, 50000, N_mock)
    A_max_values = np.random.uniform(5000, 20000, N_mock)
    district_names = [f"Mock District {i+1}" for i in range(N_mock)]


N = len(P_values) # Number of decision variables (districts)

# --- Define Bounds for x_i ---
# L_i <= x_i <= U_i
lower_bounds = np.maximum(0, delta * P_values - GA_values)
upper_bounds = A_max_values

# Ensure bounds are feasible (L_i <= U_i)
if np.any(lower_bounds > upper_bounds):
    print("Warning: Some lower bounds are greater than upper bounds. Adjusting L_i = min(L_i, U_i).")
    print("Problematic districts (L_i > U_i initially):")
    for i in range(N):
        if lower_bounds[i] > upper_bounds[i]:
            print(f"  {district_names[i]}: Original L_i={lower_bounds[i]}, U_i={upper_bounds[i]}")
    lower_bounds = np.minimum(lower_bounds, upper_bounds)


# --- Objective Function ---
def objective_function(x_vector):
    if len(x_vector) != N:
        raise ValueError(f"Input vector x_vector must have length {N}")
    if np.any(x_vector + epsilon <= 0): # Should be prevented by L_i >= 0
        return float('inf')

    term1 = alpha_prime * (P_values * C_values) / (x_vector + epsilon)
    term2 = gamma_prime * (AQ_values * x_vector) / P_values
    term3 = beta_prime * (T_values * x_vector) / P_values
    
    total_sum = np.sum(term1 - term2 - term3)
    return total_sum

# --- GENETIC ALGORITHM (Adapted from optimization_algorithms.py) ---

# Selection Methods (from your file)
def truncation_selection(fitness_values, k_truncation_selection):
    # fitness_values: array of fitness for the current population
    # k_truncation_selection: number of best individuals to select from
    sorted_indices = np.argsort(fitness_values)
    # Ensure k is not larger than population size
    k_truncation_selection = min(k_truncation_selection, len(fitness_values))
    selected_parent_indices = [sorted_indices[np.random.choice(k_truncation_selection, 2, replace=False)] for _ in fitness_values]
    return selected_parent_indices

def tournament_selection(fitness_values, k_tournament_selection):
    num_individuals = len(fitness_values)
    parents_pairs = []
    for _ in range(num_individuals // 2): # Generate pairs for (num_individuals // 2) offspring pairs
        pair = []
        for _ in range(2): # Select two parents
            candidates_indices = np.random.choice(num_individuals, k_tournament_selection, replace=False)
            best_candidate_idx_in_tournament = candidates_indices[np.argmin(fitness_values[candidates_indices])]
            pair.append(best_candidate_idx_in_tournament)
        parents_pairs.append(pair)
    # Repeat parent pairs if num_individuals is odd, or adjust logic for total parents needed
    # This structure assumes an even number of parents are selected to produce an even number of offspring.
    # The original file's `res = [p[np.random.choice(k,2)] for _ in data]` needs clarification.
    # This version creates pairs of parents.
    return parents_pairs


def roulette_wheel_selection(fitness_values):
    # Assumes minimization, so lower fitness is better.
    # Convert to maximization problem for roulette wheel (higher value = higher probability)
    max_fitness = np.max(fitness_values)
    if np.all(fitness_values == fitness_values[0]): # All fitnesses are the same
        probabilities = np.ones(len(fitness_values)) / len(fitness_values)
    else:
        adjusted_fitness = max_fitness - fitness_values + epsilon # Add epsilon to avoid zero probabilities if all fitnesses are max_fitness
        if np.sum(adjusted_fitness) == 0: # Handles case where all original fitnesses were equal (max_fitness)
             probabilities = np.ones(len(fitness_values)) / len(fitness_values)
        else:
             probabilities = adjusted_fitness / np.sum(adjusted_fitness)

    num_individuals = len(fitness_values)
    parents_pairs = []
    for _ in range(num_individuals // 2): # Generate pairs
         parents_indices = np.random.choice(num_individuals, 2, p=probabilities, replace=True) # Allow replacement for roulette
         parents_pairs.append(parents_indices)
    return parents_pairs

# Crossover Methods (from your file)
def single_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1)) # Ensure point is not 0 or len-1 for actual crossover
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def two_point_crossover(parent1, parent2):
    points = np.sort(np.random.choice(len(parent1), 2, replace=False))
    p1, p2 = points[0], points[1]
    
    child1_p1 = parent1[:p1]
    child1_p2 = parent2[p1:p2]
    child1_p3 = parent1[p2:]
    child1 = np.concatenate((child1_p1, child1_p2, child1_p3))
    
    child2_p1 = parent2[:p1]
    child2_p2 = parent1[p1:p2]
    child2_p3 = parent2[p2:]
    child2 = np.concatenate((child2_p1, child2_p2, child2_p3))
    return child1, child2

# Mutation Methods (simplified and adapted for this context)
def uniform_mutation_operator(individual, gene_index_to_mutate, low_b, high_b):
    """Mutates a single gene to a new uniform random value within its bounds."""
    mutated_individual = individual.copy()
    mutated_individual[gene_index_to_mutate] = np.random.uniform(low_b, high_b)
    return mutated_individual

def gaussian_mutation_operator(individual, gene_index_to_mutate, mu, sigma):
    """Applies Gaussian mutation to a single gene."""
    mutated_individual = individual.copy()
    mutated_individual[gene_index_to_mutate] += np.random.normal(mu, sigma)
    # Clipping will be done outside this specific operator
    return mutated_individual

# Main GA Function
def genetic_algorithm(f_obj, pop_size, num_dimensions, k_max_generations,
                      selection_method_func, k_selection_param, # k_selection_param is for truncation/tournament
                      crossover_method_func, p_crossover,
                      mutation_prob_per_gene, mutation_sigma_factor, # mutation_sigma_factor for Gaussian: sigma = factor * (U-L)
                      seed_val, l_bounds, u_bounds):
    np.random.seed(seed_val)
    
    # Initialize population
    population = np.array([np.random.uniform(l_bounds, u_bounds, num_dimensions) for _ in range(pop_size)])
    best_overall_solution = population[0].copy()
    best_overall_fitness = f_obj(best_overall_solution)

    history_best_fitness = []

    for generation in range(k_max_generations):
        # Evaluate fitness
        fitness_values = np.array([f_obj(ind) for ind in population])

        # Update best overall solution
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_overall_fitness:
            best_overall_fitness = fitness_values[current_best_idx]
            best_overall_solution = population[current_best_idx].copy()
        history_best_fitness.append(best_overall_fitness)

        # Selection
        if selection_method_func == roulette_wheel_selection:
            parent_pairs_indices = selection_method_func(fitness_values)
        else: # Truncation or Tournament
            parent_pairs_indices = selection_method_func(fitness_values, k_selection_param)
            
        new_population = np.zeros_like(population)
        
        for i in range(len(parent_pairs_indices)):
            idx1, idx2 = parent_pairs_indices[i]
            parent1, parent2 = population[idx1], population[idx2]

            # Crossover
            if np.random.rand() < p_crossover:
                child1, child2 = crossover_method_func(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            for child in [child1, child2]:
                for j in range(num_dimensions): # Per-gene mutation
                    if np.random.rand() < mutation_prob_per_gene:
                        # Using Gaussian mutation as an example here, can be swapped
                        sigma = mutation_sigma_factor * (u_bounds[j] - l_bounds[j]) # Scale sigma to range
                        child[j] += np.random.normal(0, sigma if sigma > 0 else 0.01) # ensure sigma is positive
                        # child = uniform_mutation_operator(child, j, l_bounds[j], u_bounds[j]) # Alternative
                
                # Clipping after mutation (and crossover)
                np.clip(child, l_bounds, u_bounds, out=child)

            if 2*i < pop_size:
                 new_population[2*i] = child1
            if 2*i + 1 < pop_size:
                 new_population[2*i+1] = child2
        
        population = new_population
        if (generation + 1) % (k_max_generations // 10) == 0:
            print(f"GA Generation {generation+1}/{k_max_generations}, Best Fitness: {best_overall_fitness:.4f}")
            
    return best_overall_solution, best_overall_fitness, history_best_fitness


# --- PARTICLE SWARM OPTIMIZATION (Adapted from optimization_algorithms.py) ---
class Particle:
    def __init__(self, num_dimensions, l_bounds, u_bounds):
        self.x = np.random.uniform(l_bounds, u_bounds, num_dimensions)
        # Initialize velocity (e.g., fraction of domain range or zeros)
        v_range = (u_bounds - l_bounds) * 0.1 # Max velocity as 10% of range
        self.v = np.random.uniform(-v_range, v_range, num_dimensions)
        self.x_best = self.x.copy()
        self.y_best = float('inf') # Fitness of this particle's best position

def particle_swarm_optimization(f_obj, num_particles, num_dimensions, k_max_iterations,
                                w_inertia, c1_cognitive, c2_social,
                                seed_val, l_bounds, u_bounds):
    np.random.seed(seed_val)

    # Initialize population (swarm)
    swarm = [Particle(num_dimensions, l_bounds, u_bounds) for _ in range(num_particles)]

    g_best_x = None
    g_best_y = float('inf') # Global best fitness

    # Evaluate initial population and find initial global best
    for p in swarm:
        p.y_best = f_obj(p.x)
        if p.y_best < g_best_y:
            g_best_y = p.y_best
            g_best_x = p.x.copy()
    
    history_best_fitness = []
    
    # Max velocity for clamping (e.g., 20% of the search range for each dimension)
    v_max = (u_bounds - l_bounds) * 0.2
    v_min = -v_max

    for k in range(k_max_iterations):
        for p in swarm:
            # Update velocity
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            
            cognitive_component = c1_cognitive * r1 * (p.x_best - p.x)
            social_component = c2_social * r2 * (g_best_x - p.x)
            p.v = w_inertia * p.v + cognitive_component + social_component
            
            # Clamp velocity
            np.clip(p.v, v_min, v_max, out=p.v)
            
            # Update position
            p.x += p.v
            
            # Clamp position to bounds
            np.clip(p.x, l_bounds, u_bounds, out=p.x)
            
            # Evaluate new position
            current_y = f_obj(p.x)
            
            # Update particle's best
            if current_y < p.y_best:
                p.y_best = current_y
                p.x_best = p.x.copy()
            
            # Update global best
            if current_y < g_best_y:
                g_best_y = current_y
                g_best_x = p.x.copy()
        
        history_best_fitness.append(g_best_y)
        if (k + 1) % (k_max_iterations // 10) == 0:
            print(f"PSO Iteration {k+1}/{k_max_iterations}, Best Fitness: {g_best_y:.4f}")
            
    return g_best_x, g_best_y, history_best_fitness


# --- Main Execution ---
if __name__ == "__main__":
    print("Objective Function and Optimization Setup")
    print(f"Number of districts (N): {N}")
    # print(f"Lower bounds (L_i): {lower_bounds}")
    # print(f"Upper bounds (U_i): {upper_bounds}")

    # --- Parameters for Optimization Algorithms ---
    ga_pop_size = 60
    ga_generations = 200 # Increased generations
    ga_p_crossover = 0.85
    ga_mutation_prob_per_gene = 0.05 # Mutation probability for each gene
    ga_mutation_sigma_factor = 0.1 # For Gaussian mutation: sigma = factor * (U_i - L_i)
    ga_k_selection_param = 5 # For tournament/truncation selection (e.g., tournament size)
    
    pso_num_particles = 50
    pso_iterations = 200 # Increased iterations
    pso_w_inertia = 0.7
    pso_c1_cognitive = 1.5
    pso_c2_social = 1.5
    
    random_seed = 42

    # --- Run Genetic Algorithm ---
    print("\n--- Running Genetic Algorithm ---")
    ga_best_solution, ga_best_fitness, ga_history = genetic_algorithm(
        f_obj=objective_function,
        pop_size=ga_pop_size,
        num_dimensions=N,
        k_max_generations=ga_generations,
        selection_method_func=tournament_selection, # Options: truncation_selection, tournament_selection, roulette_wheel_selection
        k_selection_param=ga_k_selection_param, 
        crossover_method_func=two_point_crossover, # Options: single_point_crossover, two_point_crossover
        p_crossover=ga_p_crossover,
        mutation_prob_per_gene=ga_mutation_prob_per_gene,
        mutation_sigma_factor=ga_mutation_sigma_factor, # For Gaussian mutation
        seed_val=random_seed,
        l_bounds=lower_bounds,
        u_bounds=upper_bounds
    )
    print("\nGA Final Results:")
    print(f"Best Solution (x_i values): {np.round(ga_best_solution, 2)}")
    print(f"Best Objective Value: {ga_best_fitness:.4f}")
    for i in range(N):
        print(f"  {district_names[i]}: Allocate {ga_best_solution[i]:.2f} (Bounds: [{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}])")


    # --- Run Particle Swarm Optimization ---
    print("\n--- Running Particle Swarm Optimization ---")
    pso_best_solution, pso_best_fitness, pso_history = particle_swarm_optimization(
        f_obj=objective_function,
        num_particles=pso_num_particles,
        num_dimensions=N,
        k_max_iterations=pso_iterations,
        w_inertia=pso_w_inertia,
        c1_cognitive=pso_c1_cognitive,
        c2_social=pso_c2_social,
        seed_val=random_seed,
        l_bounds=lower_bounds,
        u_bounds=upper_bounds
    )
    print("\nPSO Final Results:")
    print(f"Best Solution (x_i values): {np.round(pso_best_solution, 2)}")
    print(f"Best Objective Value: {pso_best_fitness:.4f}")
    for i in range(N):
        print(f"  {district_names[i]}: Allocate {pso_best_solution[i]:.2f} (Bounds: [{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}])")

    # --- Plotting Convergence (Optional) ---
    plt.figure(figsize=(12, 6))
    plt.plot(ga_history, label="GA Best Fitness")
    plt.plot(pso_history, label="PSO Best Fitness")
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Objective Function Value")
    plt.title("Optimization Algorithm Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()
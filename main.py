import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os # Added for dummy data path creation

# Parameters (default, can be overridden by scenarios)
# W1 = 0.5 # Default W1, will be set by scenario
# W2 = 0.5 # Default W2, will be set by scenario
TOTAL_LIMIT = 3_000_000
PER_TOWN_LIMIT = 1_000_000

POP_SIZE = 100  # GA Population size
GENS = 200      # GA Generations
MUT_PROB = 0.2
ELITISM = 0.1

PSO_POP_SIZE = 30 # PSO Population size (number of particles)
PSO_ITERATIONS = 100 # PSO Iterations (k_max)

def init_population(bounds):
    return [repair_individual(np.array([random.uniform(lb, ub) for lb, ub in bounds]), bounds) for _ in range(POP_SIZE)]

def mutate(individual, bounds):
    mutant = np.array([
        min(ub, max(lb, gene + np.random.normal(0, (ub - lb) * 0.1))) # Mutation scaled by bound range
        for gene, (lb, ub) in zip(individual, bounds)
    ])
    return repair_individual(mutant, bounds)

def crossover(p1, p2):
    alpha = np.random.rand(len(p1))
    child = alpha * p1 + (1 - alpha) * p2
    return child

def select(population, fitnesses):
    num_elites = max(1, int(ELITISM * POP_SIZE)) if POP_SIZE > 0 and ELITISM > 0 else 0
    if not population:
        return []
    
    # Handle non-finite fitness values before sorting
    finite_fitness_indices = [i for i, f in enumerate(fitnesses) if np.isfinite(f)]
    if not finite_fitness_indices: # All fitnesses are non-finite
        # Fallback: if no finite fitnesses, select randomly or return empty
        # This case should ideally be prevented by robust objective/repair functions
        return [random.choice(population) for _ in range(num_elites)] if population and num_elites > 0 else []


    # Sort based on finite fitnesses, placing non-finite ones effectively at the end (worst)
    # Create a list of (fitness, original_index) tuples for sorting
    sortable_fitnesses = []
    for i, f in enumerate(fitnesses):
        if np.isfinite(f):
            sortable_fitnesses.append((f, i))
        else:
            sortable_fitnesses.append((float('inf'), i)) # Treat non-finite as worst

    sorted_indices = [i for f, i in sorted(sortable_fitnesses, key=lambda x: x[0])]
    return [population[i] for i in sorted_indices[:num_elites]]


def repair_individual(x, bounds):
    x = np.array([np.clip(val, bounds[i][0], bounds[i][1]) for i, val in enumerate(x)])
    total = np.sum(x)
    if total > TOTAL_LIMIT:
        # Avoid division by zero if total is zero (though unlikely if TOTAL_LIMIT > 0)
        if total > 1e-9:
             x = x * (TOTAL_LIMIT / total)
        else: # If total is near zero and exceeds TOTAL_LIMIT (impossible if TOTAL_LIMIT > 0)
             x = np.zeros_like(x) # Or handle as an error/special case


    x = np.array([np.clip(val, bounds[i][0], bounds[i][1]) for i, val in enumerate(x)])
    if np.sum(x) > TOTAL_LIMIT:
        current_sum = np.sum(x)
        if current_sum > 1e-9: # Avoid division by zero
            x = x * (TOTAL_LIMIT / current_sum)
        else:
            x = np.zeros_like(x)
    return x


def run_ga(bounds, objective_func_ga): # Renamed to avoid conflict
    population = init_population(bounds)
    history = []

    for gen in range(GENS):
        fitnesses = [objective_func_ga(ind) for ind in population]
        
        if not fitnesses:
            best_fitness_this_gen = float('inf')
        else:
            valid_fitnesses = [f for f in fitnesses if np.isfinite(f)]
            if not valid_fitnesses:
                best_fitness_this_gen = float('inf')
            else:
                best_fitness_this_gen = min(valid_fitnesses)
        
        history.append(best_fitness_this_gen)
        selected_elites = select(population, fitnesses)
        
        if not selected_elites and POP_SIZE > 0 :
            if population and any(np.isfinite(f) for f in fitnesses): # Check if any finite fitness exists
                 # Get indices of finite fitnesses
                 finite_indices = [i for i, f_val in enumerate(fitnesses) if np.isfinite(f_val)]
                 # Get the index of the minimum finite fitness
                 min_finite_idx = finite_indices[np.argmin([fitnesses[i] for i in finite_indices])]
                 selected_elites = [population[min_finite_idx]]
            else:
                 selected_elites = [init_population(bounds)[0] for _ in range(max(1, int(ELITISM*POP_SIZE)))]

        new_pop = list(selected_elites)

        while len(new_pop) < POP_SIZE:
            if len(selected_elites) >= 2:
                parents = random.sample(selected_elites, 2)
            elif selected_elites:
                parents = [selected_elites[0], random.choice(population)]
            else: 
                parents = random.sample(population, 2) if len(population) >= 2 else [init_population(bounds)[0]]*2


            child1 = crossover(parents[0], parents[1])
            
            if random.random() < MUT_PROB:
                child1 = mutate(child1, bounds)
            
            child1 = repair_individual(child1, bounds)
            new_pop.append(child1)
        population = new_pop

    final_fitnesses = [objective_func_ga(ind) for ind in population]
    
    best_idx = -1
    current_best_fitness = float('inf')
    for i, f in enumerate(final_fitnesses):
        if np.isfinite(f) and f < current_best_fitness:
            current_best_fitness = f
            best_idx = i
            
    if best_idx != -1:
        best_individual = population[best_idx]
    else: 
        print("Warning: No valid best individual found by GA. Returning first valid or re-initialized individual.")
        # Try to find any individual with finite fitness
        valid_indices = [i for i, f_val in enumerate(final_fitnesses) if np.isfinite(f_val)]
        if valid_indices:
            best_individual = population[valid_indices[0]] # Pick the first one found
        else: # Absolute fallback
            best_individual = init_population(bounds)[0]
    
    best_score_val = objective_func_ga(best_individual)


    return best_individual, best_score_val, history

class Particle:
    def __init__(self, x, v, bounds, repair_func, objective_func_pso): # Renamed
        self.x = np.array(x)
        self.v = np.array(v)
        self.bounds = bounds
        self.repair_func = repair_func
        self.objective_func_pso = objective_func_pso # Renamed

        self.x = self.repair_func(self.x, self.bounds)
        self.pbest_x = self.x.copy()
        self.pbest_score = self.objective_func_pso(self.pbest_x) # Use renamed
    
    def update_velocity(self, gbest_x, w, c1, c2, n_dim):
        r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
        self.v = w * self.v + \
                   c1 * r1 * (self.pbest_x - self.x) + \
                   c2 * r2 * (gbest_x - self.x)
        max_vel_ratio = 0.2 
        for i in range(n_dim):
            max_v = (self.bounds[i][1] - self.bounds[i][0]) * max_vel_ratio
            self.v[i] = np.clip(self.v[i], -max_v, max_v)

    def update_position(self):
        self.x += self.v
        self.x = self.repair_func(self.x, self.bounds)
        current_score = self.objective_func_pso(self.x) # Use renamed
        if np.isfinite(current_score) and current_score < self.pbest_score:
            self.pbest_x = self.x.copy()
            self.pbest_score = current_score


def particle_swarm_optimization(objective_func_pso, population_particles, k_max, w=0.7, c1=1.5, c2=1.5): # Renamed
    if not population_particles:
        return None, float('inf'), []

    gbest_x = None
    gbest_score = float('inf')
    
    # Initialize global best from particles, ensuring finite scores are preferred
    for particle in population_particles:
        if np.isfinite(particle.pbest_score) and particle.pbest_score < gbest_score:
            gbest_score = particle.pbest_score
            gbest_x = particle.pbest_x.copy()

    # If no particle had a finite pbest_score initially, gbest_x might still be None
    if gbest_x is None and population_particles:
         # Fallback: pick the first particle's pbest if available, or initialize
         if population_particles[0].pbest_x is not None:
            gbest_x = population_particles[0].pbest_x.copy()
            gbest_score = population_particles[0].pbest_score # Might be inf
         else: # Should not happen if particle init is correct
            return None, float('inf'), []


    if gbest_x is None: # Still no gbest_x means population_particles was empty or problematic
        return None, float('inf'), []
            
    n_dim = len(gbest_x)
    history = []

    for k in range(k_max):
        for particle in population_particles:
            particle.update_velocity(gbest_x, w, c1, c2, n_dim)
            particle.update_position() # This updates particle.pbest_score
            
            if np.isfinite(particle.pbest_score) and particle.pbest_score < gbest_score:
                gbest_score = particle.pbest_score
                gbest_x = particle.pbest_x.copy()
        history.append(gbest_score)

    return gbest_x, gbest_score, history


def create_pso_population(num_particles, bounds, repair_func, objective_func_pso, seed): # Renamed
    # np.random.seed(seed) # Seeding here might make all initial particles too similar if objective_func is deterministic
    # random.seed(seed) # Better to seed once globally if needed
    population = []
    n_dim = len(bounds)
    for _ in range(num_particles):
        x = np.array([random.uniform(lb, ub) for lb, ub in bounds])
        v = np.array([random.uniform(-(bounds[i][1]-bounds[i][0])*0.1, (bounds[i][1]-bounds[i][0])*0.1) for i in range(n_dim)])
        
        # Ensure objective_func_pso is callable for particle initialization
        particle = Particle(x, v, bounds, repair_func, objective_func_pso)
        population.append(particle)
    return population


def main():
    # Define W1 and W2 scenarios
    weight_scenarios = [
        {"W1": 0.2, "W2": 0.8, "label": "W1_0.2_W2_0.8"},
        {"W1": 0.5, "W2": 0.5, "label": "W1_0.5_W2_0.5"},
        {"W1": 0.8, "W2": 0.2, "label": "W1_0.8_W2_0.2"},
    ]

    df_base = pd.read_csv("result/birlesik_ilce_verisi.csv")
    df_results = df_base.copy()

    # Toplu taşıma puanı (calculated once)
    df_base["Ti_raw"] = df_base["Minibus_Durak_Sayisi"] + df_base["Taksi_Durak_Sayisi"] + 2 * df_base["Rayli_Istasyon_Sayisi"]

    # Normalizations (independent of W1, W2 - calculated once)
    GA_real = df_base["alan_metrekare"].values.astype(float)
    GA_min, GA_max = GA_real.min(), GA_real.max()
    GA_norm = (GA_real - GA_min) / (GA_max - GA_min + 1e-9)

    P_real = df_base["Nufus"].values.astype(float)
    P_min, P_max = P_real.min(), P_real.max()
    P_norm = (P_real - P_min) / (P_max - P_min + 1e-9)

    Ti_real = df_base["Ti_raw"].values.astype(float)
    Ti_min, Ti_max = Ti_real.min(), Ti_real.max()
    Ti_norm = (Ti_real - Ti_min) / (Ti_max - Ti_min + 1e-9)

    AQI_real = df_base["Ortalama_AQI"].values.astype(float)
    AQI_min, AQI_max = AQI_real.min(), AQI_real.max()
    AQI_norm = (AQI_real - AQI_min) / (AQI_max - AQI_min + 1e-9)

    # Bounds (calculated once)
    lower_bounds = np.zeros(len(GA_real))
    upper_bounds = np.minimum(GA_real / 2, PER_TOWN_LIMIT)
    for i in range(len(upper_bounds)):
        if lower_bounds[i] > upper_bounds[i]:
            upper_bounds[i] = lower_bounds[i]
    bounds = list(zip(lower_bounds, upper_bounds))

    # Dictionary to hold parameters for the objective function that change per scenario
    objective_params = {'S_norm_current': None}

    # Define objective function (it will use S_norm_current from objective_params)
    def objective(x_new_green_area_real):
        x_new_green_area_real = np.array(x_new_green_area_real)
        x_norm = x_new_green_area_real / (GA_max + 1e-9)
        total_norm_green_area = GA_norm + x_norm
        
        current_S_norm = objective_params['S_norm_current'] # Use S_norm from the dictionary
        
        performance_term = (current_S_norm * P_norm) / (total_norm_green_area + 1e-6)
        base = np.sum(performance_term)

        current_total_green_area_real = GA_real + x_new_green_area_real
        green_per_person_real = current_total_green_area_real / (P_real + 1e-9)
        
        if np.max(green_per_person_real) > 1e-9 :
            normalized_gpp = green_per_person_real / np.max(green_per_person_real)
        else:
            normalized_gpp = np.zeros_like(green_per_person_real)

        initial_green_per_person_real = GA_real / (P_real + 1e-9)
        if np.max(initial_green_per_person_real) > 1e-9:
             norm_initial_gpp = initial_green_per_person_real / np.max(initial_green_per_person_real)
        else:
            norm_initial_gpp = np.zeros_like(initial_green_per_person_real)

        fairness_penalty = np.sum(x_new_green_area_real * norm_initial_gpp)
        penalty_coefficient = 0.000005
        return base + penalty_coefficient * fairness_penalty

    # Loop through scenarios
    for scenario in weight_scenarios:
        W1_scen = scenario["W1"]
        W2_scen = scenario["W2"]
        label = scenario["label"]

        print(f"\n--- Running Scenario: {label} (W1={W1_scen}, W2={W2_scen}) ---")

        # Calculate S_norm for the current scenario and update objective_params
        objective_params['S_norm_current'] = W1_scen * AQI_norm + W2_scen * Ti_norm

        print("🧬 Running Genetic Algorithm...")
        # Pass the 'objective' function which now uses the updated 'S_norm_current'
        best_solution_ga, best_score_ga, history_ga = run_ga(bounds, objective)

        print("\n⚙️ Running Particle Swarm Optimization...")
        # Seed for PSO population creation to ensure some consistency if desired for PSO part
        # Note: Seeding inside create_pso_population might be better if full reproducibility per call is needed
        # For now, using a fixed seed for each scenario's PSO population creation
        pso_particles = create_pso_population(PSO_POP_SIZE, bounds, repair_individual, objective, seed=42)
        best_solution_pso, best_score_pso, history_pso = particle_swarm_optimization(
            objective_func_pso=objective, # PSO also uses the same objective
            population_particles=pso_particles,
            k_max=PSO_ITERATIONS
        )

        # Store results with scenario-specific column names
        ga_col_yeni = f"Yeni_Yapilacak_Yesil_Alan_GA_{label}"
        ga_col_toplam = f"Toplam_Yesil_Alan_GA_{label}"
        pso_col_yeni = f"Yeni_Yapilacak_Yesil_Alan_PSO_{label}"
        pso_col_toplam = f"Toplam_Yesil_Alan_PSO_{label}"

        df_results[ga_col_yeni] = best_solution_ga
        df_results[ga_col_toplam] = df_results["alan_metrekare"] + best_solution_ga
        
        if best_solution_pso is not None:
            df_results[pso_col_yeni] = best_solution_pso
            df_results[pso_col_toplam] = df_results["alan_metrekare"] + best_solution_pso
        else:
            df_results[pso_col_yeni] = 0 # Or np.nan
            df_results[pso_col_toplam] = df_results["alan_metrekare"]

        print(f"\n--- Results for Scenario: {label} ---")
        print(f"✅ En iyi GA skoru: {best_score_ga:.4f}")
        print(f"📏 Toplam yeni yapılan yeşil alan (GA): {np.sum(best_solution_ga):,.2f} m²")
        
        if best_solution_pso is not None:
            print(f"✅ En iyi PSO skoru: {best_score_pso:.4f}")
            print(f"📏 Toplam yeni yapılan yeşil alan (PSO): {np.sum(best_solution_pso):,.2f} m²")
        else:
            print("⚠️ PSO did not return a valid solution for this scenario.")

    # Save the consolidated DataFrame
    output_filename = "optimum_yesil_alan_sonuclari_SCENARIOS.csv"
    df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n📁 Tüm senaryo sonuçları '{output_filename}' dosyasına kaydedildi.")

    # Convergence plots are not shown per scenario to avoid multiple pop-ups.
    # If needed, history_ga and history_pso for the last scenario (or all stored ones) can be plotted here.
    # For example, to plot the last GA history:
    # if 'history_ga' in locals() and history_ga: # Check if history_ga exists and is not empty
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(history_ga, marker='o', linestyle='-', color='green')
    #     plt.title(f"GA Convergence for last scenario ({label})")
    #     plt.xlabel("Nesil")
    #     plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    # Create a dummy CSV for testing if it doesn't exist
    dummy_csv_path = "result/birlesik_ilce_verisi.csv"
    try:
        pd.read_csv(dummy_csv_path)
    except FileNotFoundError:
        print(f"Dummy '{dummy_csv_path}' not found. Creating one for testing.")
        if not os.path.exists("result"):
            os.makedirs("result")
        dummy_data = {
            "ILCE": [f"Ilce_{i}" for i in range(10)],
            "Minibus_Durak_Sayisi": np.random.randint(5, 50, 10),
            "Taksi_Durak_Sayisi": np.random.randint(2, 20, 10),
            "Rayli_Istasyon_Sayisi": np.random.randint(0, 5, 10),
            "alan_metrekare": np.random.randint(100000, 2000000, 10).astype(float),
            "Nufus": np.random.randint(50000, 500000, 10).astype(float),
            "Ortalama_AQI": np.random.uniform(20, 80, 10)
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(dummy_csv_path, index=False)
        print("Dummy CSV created.")
    
    # It's good practice to seed random number generators for reproducibility if needed
    # random.seed(42)
    # np.random.seed(42)
    main()
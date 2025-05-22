import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Parametreler
W1 = 0.5
W2 = 0.5
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
    # Ensure ELITISM * POP_SIZE is at least 1 if POP_SIZE > 0 and ELITISM > 0
    num_elites = max(1, int(ELITISM * POP_SIZE)) if POP_SIZE > 0 and ELITISM > 0 else 0
    if not population: # Handle empty population case
        return []
    idx = np.argsort(fitnesses)
    return [population[i] for i in idx[:num_elites]]


def repair_individual(x, bounds):
    # Apply per-town limits (clipping to individual bounds)
    x = np.array([np.clip(val, bounds[i][0], bounds[i][1]) for i, val in enumerate(x)])

    # Apply total limit
    total = np.sum(x)
    if total > TOTAL_LIMIT:
        x = x * (TOTAL_LIMIT / total)
    # Ensure individual elements are still within their bounds after scaling for total limit
    x = np.array([np.clip(val, bounds[i][0], bounds[i][1]) for i, val in enumerate(x)])
    # It's possible that the sum might slightly exceed TOTAL_LIMIT after the second clip if many values hit their upper bound.
    # A more robust repair might iterate or use a different scaling, but this is usually sufficient.
    # For strict adherence, a final check and scale could be added:
    if np.sum(x) > TOTAL_LIMIT:
        # Fallback: if still over, scale down again. This should be rare.
        x = x * (TOTAL_LIMIT / np.sum(x))
    return x


def run_ga(bounds, objective_func):
    population = init_population(bounds)
    history = []

    for gen in range(GENS):
        fitnesses = [objective_func(ind) for ind in population]
        
        # Handle cases where fitnesses might be empty or contain non-finite values
        if not fitnesses:
            print(f"Warning: Empty fitnesses list in generation {gen}.")
            # Potentially re-initialize population or handle error
            best_fitness_this_gen = float('inf') # Or some other placeholder
        else:
            valid_fitnesses = [f for f in fitnesses if np.isfinite(f)]
            if not valid_fitnesses:
                print(f"Warning: No finite fitness values in generation {gen}.")
                best_fitness_this_gen = float('inf')
            else:
                best_fitness_this_gen = min(valid_fitnesses)
        
        history.append(best_fitness_this_gen)

        selected_elites = select(population, fitnesses)
        
        # Ensure selected_elites is not empty for sampling if POP_SIZE > 0
        if not selected_elites and POP_SIZE > 0 :
             # Fallback: if no elites (e.g. all fitnesses were inf), re-init or pick randomly
            print(f"Warning: No elites selected in generation {gen}. Taking best from current population if possible.")
            if population and all(np.isfinite(f) for f in fitnesses):
                 selected_elites = [population[np.argmin(fitnesses)]]
            else: # If all else fails, re-initialize a small pool of elites
                 selected_elites = [init_population(bounds)[0] for _ in range(max(1, int(ELITISM*POP_SIZE)))]


        new_pop = list(selected_elites) # Start new population with elites

        while len(new_pop) < POP_SIZE:
            if len(selected_elites) >= 2:
                parents = random.sample(selected_elites, 2)
            elif selected_elites: # If only one elite, use it twice or combine with another method
                parents = [selected_elites[0], random.choice(population)] # Or simply duplicate the elite for crossover
            else: # Should not happen if above checks are in place
                parents = random.sample(population, 2) # Fallback to sampling from the whole population

            child1 = crossover(parents[0], parents[1]) # child1 is an array
            
            if random.random() < MUT_PROB:
                child1 = mutate(child1, bounds)
            
            child1 = repair_individual(child1, bounds)
            new_pop.append(child1)
        population = new_pop

    # Final selection of the best individual
    final_fitnesses = [objective_func(ind) for ind in population]
    
    best_idx = -1
    current_best_fitness = float('inf')
    for i, f in enumerate(final_fitnesses):
        if np.isfinite(f) and f < current_best_fitness:
            current_best_fitness = f
            best_idx = i
            
    if best_idx != -1:
        best_individual = population[best_idx]
        best_score_val = current_best_fitness
    else: # Fallback if no valid best individual is found
        print("Warning: No valid best individual found by GA. Returning first individual.")
        best_individual = population[0] if population else init_population(bounds)[0] # Ensure there's something to return
        best_score_val = objective_func(best_individual)

    return best_individual, best_score_val, history

class Particle:
    def __init__(self, x, v, bounds, repair_func, objective_func):
        self.x = np.array(x)
        self.v = np.array(v)
        self.bounds = bounds # Store bounds for repair
        self.repair_func = repair_func # Store repair function
        self.objective_func = objective_func # Store objective function

        self.x = self.repair_func(self.x, self.bounds) # Initial repair
        self.pbest_x = self.x.copy()
        self.pbest_score = self.objective_func(self.pbest_x)
    
    def update_velocity(self, gbest_x, w, c1, c2, n_dim):
        r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
        self.v = w * self.v + \
                   c1 * r1 * (self.pbest_x - self.x) + \
                   c2 * r2 * (gbest_x - self.x)
        # Simple velocity clamping (can be made more sophisticated)
        # Max velocity can be a fraction of the dynamic range of variables
        max_vel_ratio = 0.2 
        for i in range(n_dim):
            max_v = (self.bounds[i][1] - self.bounds[i][0]) * max_vel_ratio
            self.v[i] = np.clip(self.v[i], -max_v, max_v)

    def update_position(self):
        self.x += self.v
        self.x = self.repair_func(self.x, self.bounds) # Repair after position update
        current_score = self.objective_func(self.x)
        if current_score < self.pbest_score:
            self.pbest_x = self.x.copy()
            self.pbest_score = current_score


def particle_swarm_optimization(objective_func, population_particles, k_max, w=0.7, c1=1.5, c2=1.5):
    if not population_particles:
        return None, float('inf'), []

    # Initialize global best
    gbest_x = population_particles[0].pbest_x.copy()
    gbest_score = population_particles[0].pbest_score
    
    for particle in population_particles:
        if particle.pbest_score < gbest_score:
            gbest_score = particle.pbest_score
            gbest_x = particle.pbest_x.copy()
            
    n_dim = len(gbest_x)
    history = []

    for k in range(k_max):
        for particle in population_particles:
            particle.update_velocity(gbest_x, w, c1, c2, n_dim)
            particle.update_position()
            
            if particle.pbest_score < gbest_score:
                gbest_score = particle.pbest_score
                gbest_x = particle.pbest_x.copy()
        history.append(gbest_score)
        # print(f"PSO Iteration {k+1}/{k_max}, Best Score: {gbest_score:.2f}")


    return gbest_x, gbest_score, history


def create_pso_population(num_particles, bounds, repair_func, objective_func, seed):
    np.random.seed(seed)
    population = []
    n_dim = len(bounds)
    for _ in range(num_particles):
        # Initial position within bounds
        x = np.array([random.uniform(lb, ub) for lb, ub in bounds])
        # Initial velocity (can be small random values or zeros)
        # A common approach is to initialize velocities to a small fraction of the variable range
        v = np.array([random.uniform(-(bounds[i][1]-bounds[i][0])*0.1, (bounds[i][1]-bounds[i][0])*0.1) for i in range(n_dim)])
        
        particle = Particle(x, v, bounds, repair_func, objective_func)
        population.append(particle)
    return population


def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")

    # Toplu taşıma puanı
    df["Ti_raw"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]

    # Normalizasyonlar
    GA_real = df["alan_metrekare"].values.astype(float) # Ensure float for calculations
    GA_min, GA_max = GA_real.min(), GA_real.max()
    # Add epsilon to prevent division by zero if min == max
    GA_norm = (GA_real - GA_min) / (GA_max - GA_min + 1e-9)


    P_real = df["Nufus"].values.astype(float)
    P_min, P_max = P_real.min(), P_real.max()
    P_norm = (P_real - P_min) / (P_max - P_min + 1e-9)


    Ti_real = df["Ti_raw"].values.astype(float)
    Ti_min, Ti_max = Ti_real.min(), Ti_real.max()
    Ti_norm = (Ti_real - Ti_min) / (Ti_max - Ti_min + 1e-9)


    AQI_real = df["Ortalama_AQI"].values.astype(float)
    AQI_min, AQI_max = AQI_real.min(), AQI_real.max()
    AQI_norm = (AQI_real - AQI_min) / (AQI_max - AQI_min + 1e-9)

    S_norm = W1 * AQI_norm + W2 * Ti_norm

    lower_bounds = np.zeros(len(GA_real))
    upper_bounds = np.minimum(GA_real / 2, PER_TOWN_LIMIT) # GA_real/2 can be zero if GA_real is zero
    # Ensure lower_bounds[i] <= upper_bounds[i]
    for i in range(len(upper_bounds)):
        if lower_bounds[i] > upper_bounds[i]:
            upper_bounds[i] = lower_bounds[i]

    bounds = list(zip(lower_bounds, upper_bounds))

    def objective(x_new_green_area_real): # x is the proposed NEW green area in real m^2
        # Ensure x is a numpy array for vectorized operations
        x_new_green_area_real = np.array(x_new_green_area_real)

        x_norm = x_new_green_area_real / (GA_max + 1e-9) # Normalize new area relative to max existing area

        total_norm_green_area = GA_norm + x_norm
        
        
        performance_term = (S_norm * P_norm) / (total_norm_green_area + 1e-6) # per town
        base = np.sum(performance_term) # Sum over all towns. This is to be MINIMIZED.

        # Fairness penalty (using real values for green_per_person)
        current_total_green_area_real = GA_real + x_new_green_area_real
        green_per_person_real = current_total_green_area_real / (P_real + 1e-9) # Avoid division by zero

        # Normalize green_per_person for penalty calculation to avoid scale issues
        # This normalization should be based on the current state after adding x
        if np.max(green_per_person_real) > 1e-9 : # Avoid division by zero if all gpp are zero
            normalized_gpp = green_per_person_real / np.max(green_per_person_real)
        else:
            normalized_gpp = np.zeros_like(green_per_person_real)

        # Penalty: if a town *already* has high green_per_person (before adding x)
        # and we are *still* adding green space (x > 0), penalize.
        # A better penalty might consider *initial* green_per_person.
        # Let's use initial green area per person for the penalty's weighting factor.
        initial_green_per_person_real = GA_real / (P_real + 1e-9)
        if np.max(initial_green_per_person_real) > 1e-9:
             norm_initial_gpp = initial_green_per_person_real / np.max(initial_green_per_person_real)
        else:
            norm_initial_gpp = np.zeros_like(initial_green_per_person_real)

        # Penalize adding new green space (x_new_green_area_real) to towns that *already* have a high normalized initial GPP.
        fairness_penalty = np.sum(x_new_green_area_real * norm_initial_gpp)
        
        # The objective is to MINIMIZE this combined value
        penalty_coefficient = 0.000005 # Adjusted penalty coefficient
        return base + penalty_coefficient * fairness_penalty


    print("🧬 Running Genetic Algorithm...")
    best_solution_ga, best_score_ga, history_ga = run_ga(bounds, objective)

    print("\n Running Particle Swarm Optimization...")
    # Create population for PSO
    pso_particles = create_pso_population(PSO_POP_SIZE, bounds, repair_individual, objective, seed=42)
    best_solution_pso, best_score_pso, history_pso = particle_swarm_optimization(
        objective_func=objective,
        population_particles=pso_particles,
        k_max=PSO_ITERATIONS
    )

    df["Yeni_Yapilacak_Yesil_Alan_GA"] = best_solution_ga
    df["Toplam_Yesil_Alan_GA"] = df["alan_metrekare"] + best_solution_ga
    
    if best_solution_pso is not None:
        df["Yeni_Yapilacak_Yesil_Alan_PSO"] = best_solution_pso
        df["Toplam_Yesil_Alan_PSO"] = df["alan_metrekare"] + best_solution_pso
    else:
        df["Yeni_Yapilacak_Yesil_Alan_PSO"] = 0 # Or np.nan
        df["Toplam_Yesil_Alan_PSO"] = df["alan_metrekare"]


    print(f"\n--- Genetic Algorithm Results ---")
    print(f"✅ En iyi GA skoru: {best_score_ga:.4f}")
    print(f"📏 Toplam yeni yapılan yeşil alan (GA): {np.sum(best_solution_ga):,.2f} m²")
    
    if best_solution_pso is not None:
        print(f"\n--- Particle Swarm Optimization Results ---")
        print(f"✅ En iyi PSO skoru: {best_score_pso:.4f}")
        print(f"📏 Toplam yeni yapılan yeşil alan (PSO): {np.sum(best_solution_pso):,.2f} m²")
    else:
        print("\n--- Particle Swarm Optimization Results ---")
        print("⚠️ PSO did not return a valid solution.")

    print("\n--- DataFrame Preview (First 5 rows with new areas) ---")
    columns_to_show = ["ILCE", "Yeni_Yapilacak_Yesil_Alan_GA", "Toplam_Yesil_Alan_GA"]
    if "Yeni_Yapilacak_Yesil_Alan_PSO" in df.columns:
        columns_to_show.extend(["Yeni_Yapilacak_Yesil_Alan_PSO", "Toplam_Yesil_Alan_PSO"])
    print(df[columns_to_show].head())

    output_filename = "optimum_yesil_alan_sonuclari.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig') # Added encoding for wider compatibility
    print(f"\n📁 Sonuçlar '{output_filename}' dosyasına kaydedildi.")

    # Plot GA Convergence
    plt.figure(figsize=(12, 7))
    plt.plot(history_ga, marker='o', linestyle='-', color='green', label='GA Best Fitness')
    plt.title("Genetik Algoritma Yakınsama Grafiği")
    plt.xlabel("Nesil")
    plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
    plt.grid(True)
    plt.tight_layout()

    # Plot PSO Convergence
    if history_pso:
        plt.figure(figsize=(12, 7))
        plt.plot(history_pso, marker='x', linestyle='--', color='blue', label='PSO Best Fitness')
        plt.title("Particle Swarm Optimization Yakınsama Grafiği")
        plt.xlabel("İterasyon")
        plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
        plt.grid(True)
        plt.tight_layout()
    
    # Optionally, show both on one plot if scales are comparable
    if history_ga and history_pso:
        plt.figure(figsize=(12,7))
        plt.plot(history_ga, marker='o', linestyle='-', color='green', label=f'GA (Best: {best_score_ga:.2f})')
        # Pad PSO history if it's shorter than GA history for combined plot, or plot against iterations
        iterations_pso = np.arange(len(history_pso))
        generations_ga = np.arange(len(history_ga))
        
        plt.plot(iterations_pso, history_pso, marker='x', linestyle='--', color='blue', label=f'PSO (Best: {best_score_pso:.2f})')
        plt.title("GA vs PSO Yakınsama")
        plt.xlabel("İterasyon / Nesil")
        plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
        # Consider log scale for y-axis if values change drastically
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create a dummy CSV for testing if it doesn't exist
    try:
        pd.read_csv("result/birlesik_ilce_verisi.csv")
    except FileNotFoundError:
        print("Dummy 'result/birlesik_ilce_verisi.csv' not found. Creating one for testing.")
        import os
        if not os.path.exists("result"):
            os.makedirs("result")
        dummy_data = {
            "ILCE": [f"Ilce_{i}" for i in range(10)],
            "Minibus_Durak_Sayisi": np.random.randint(5, 50, 10),
            "Taksi_Durak_Sayisi": np.random.randint(2, 20, 10),
            "Rayli_Istasyon_Sayisi": np.random.randint(0, 5, 10),
            "alan_metrekare": np.random.randint(100000, 2000000, 10),
            "Nufus": np.random.randint(50000, 500000, 10),
            "Ortalama_AQI": np.random.uniform(20, 80, 10)
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv("result/birlesik_ilce_verisi.csv", index=False)
        print("Dummy CSV created.")
        
    main()
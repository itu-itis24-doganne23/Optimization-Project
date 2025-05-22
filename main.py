import pandas as pd
import numpy as np
import random

# Parametreler
W1 = 0.5
W2 = 0.5
rho = 30  # kişi başı maksimum yeşil alan
TOTAL_LIMIT = 1_000_000  # toplam yapılabilecek yeşil alan sınırı

POP_SIZE = 100
GENS = 200
MUT_PROB = 0.2
ELITISM = 0.1

### GA Fonksiyonları ###
def init_population(bounds):
    return [np.array([random.uniform(lb, ub) for lb, ub in bounds]) for _ in range(POP_SIZE)]

def mutate(individual, bounds):
    return np.array([
        min(ub, max(lb, gene + np.random.normal(0, 1000)))
        for gene, (lb, ub) in zip(individual, bounds)
    ])

def crossover(p1, p2):
    alpha = np.random.rand(len(p1))
    return alpha * p1 + (1 - alpha) * p2

def select(population, fitnesses):
    idx = np.argsort(fitnesses)
    return [population[i] for i in idx[:int(ELITISM * POP_SIZE)]]

def run_ga(bounds, objective):
    population = init_population(bounds)
    for _ in range(GENS):
        fitnesses = [objective(ind) for ind in population]
        new_pop = select(population, fitnesses)
        while len(new_pop) < POP_SIZE:
            parents = random.sample(new_pop, 2)
            child = crossover(parents[0], parents[1])
            if random.random() < MUT_PROB:
                child = mutate(child, bounds)
            new_pop.append(child)
        population = new_pop
    best = min(population, key=objective)
    return best, objective(best)

### PSO Sınıfı ###
class Particle:
    def __init__(self, num_dimensions, l_bounds, u_bounds):
        self.x = np.random.uniform(l_bounds, u_bounds, num_dimensions)
        v_range = (u_bounds - l_bounds) * 0.1
        self.v = np.random.uniform(-v_range, v_range, num_dimensions)
        self.x_best = self.x.copy()
        self.y_best = float('inf')

def particle_swarm_optimization(f_obj, num_particles, num_dimensions, k_max_iterations,
                                w_inertia, c1_cognitive, c2_social,
                                seed_val, l_bounds, u_bounds):
    np.random.seed(seed_val)
    swarm = [Particle(num_dimensions, l_bounds, u_bounds) for _ in range(num_particles)]

    g_best_x = None
    g_best_y = float('inf')
    v_max = (u_bounds - l_bounds) * 0.2
    v_min = -v_max
    history_best_fitness = []

    for p in swarm:
        p.y_best = f_obj(p.x)
        if p.y_best < g_best_y:
            g_best_y = p.y_best
            g_best_x = p.x.copy()

    for k in range(k_max_iterations):
        for p in swarm:
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            cognitive = c1_cognitive * r1 * (p.x_best - p.x)
            social = c2_social * r2 * (g_best_x - p.x)
            p.v = w_inertia * p.v + cognitive + social
            np.clip(p.v, v_min, v_max, out=p.v)
            p.x += p.v
            np.clip(p.x, l_bounds, u_bounds, out=p.x)

            current_y = f_obj(p.x)
            if current_y < p.y_best:
                p.y_best = current_y
                p.x_best = p.x.copy()
            if current_y < g_best_y:
                g_best_y = current_y
                g_best_x = p.x.copy()
        history_best_fitness.append(g_best_y)

    return g_best_x, g_best_y, history_best_fitness

### Ana Fonksiyon ###
def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")

    df["Ti"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]
    df["Si"] = W1 * df["Ortalama_AQI"] + W2 * df["Ti"]

    GA = df["alan_metrekare"].values
    P = df["Nufus"].values
    S = df["Si"].values

    lower_bounds = np.zeros(len(GA))
    upper_bounds = np.minimum(rho * P - GA, GA / 2)
    upper_bounds = np.maximum(upper_bounds, 0)  # negatif olmasın

    # Geçerli ilçeleri filtrele
    valid_indices = np.where(upper_bounds > lower_bounds)[0]
    filtered_l_bounds = lower_bounds[valid_indices]
    filtered_u_bounds = upper_bounds[valid_indices]
    filtered_GA = GA[valid_indices]
    filtered_P = P[valid_indices]
    filtered_S = S[valid_indices]

    # Ortak objective
    def objective(x):
        total_new_area = np.sum(x)
        penalty = 1e6 if total_new_area > TOTAL_LIMIT else 0
        return np.sum((filtered_S * filtered_P) / (filtered_GA + x + 1)) + penalty

    # ---- GA ----
    ga_bounds = list(zip(filtered_l_bounds, filtered_u_bounds))
    best_ga, score_ga = run_ga(ga_bounds, objective)

    # ---- PSO ----
    best_pso, score_pso, _ = particle_swarm_optimization(
        objective,
        num_particles=100,
        num_dimensions=len(filtered_GA),
        k_max_iterations=200,
        w_inertia=0.7,
        c1_cognitive=1.5,
        c2_social=1.5,
        seed_val=42,
        l_bounds=filtered_l_bounds,
        u_bounds=filtered_u_bounds
    )

    # Sonuçları tam dizilere yerleştir
    full_ga = np.zeros(len(GA))
    full_ga[valid_indices] = best_ga

    full_pso = np.zeros(len(GA))
    full_pso[valid_indices] = best_pso

    df["GA_Yeni_Yesil_Alan"] = full_ga
    df["GA_Toplam_Yesil_Alan"] = df["alan_metrekare"] + full_ga

    df["PSO_Yeni_Yesil_Alan"] = full_pso
    df["PSO_Toplam_Yesil_Alan"] = df["alan_metrekare"] + full_pso

    # Sonuçları yazdır
    print(f"\n✅ GA Skoru: {score_ga:.2f} | Yeni yeşil alan: {np.sum(best_ga):,.2f} m²")
    print(f"✅ PSO Skoru: {score_pso:.2f} | Yeni yeşil alan: {np.sum(best_pso):,.2f} m²")
    print(df[["ILCE", "GA_Yeni_Yesil_Alan", "PSO_Yeni_Yesil_Alan"]])

    # CSV'ye kaydet
    df.to_csv("optimum_yesil_alan_sonuclari.csv", index=False)
    print("\n📁 Sonuçlar 'optimum_yesil_alan_sonuclari.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()

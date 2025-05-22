import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Parametreler
W1 = 0.5
W2 = 0.5
TOTAL_LIMIT = 3_000_000
PER_TOWN_LIMIT = 1_000_000

POP_SIZE = 100
GENS = 200
MUT_PROB = 0.2
ELITISM = 0.1

# GA Yardımcı Fonksiyonlar
def init_population(bounds):
    return [repair_individual(np.array([random.uniform(lb, ub) for lb, ub in bounds]), bounds) for _ in range(POP_SIZE)]

def mutate(individual, bounds):
    mutant = np.array([
        min(ub, max(lb, gene + np.random.normal(0, 1000)))
        for gene, (lb, ub) in zip(individual, bounds)
    ])
    return repair_individual(mutant, bounds)

def crossover(p1, p2):
    alpha = np.random.rand(len(p1))
    child = alpha * p1 + (1 - alpha) * p2
    return child

def select(population, fitnesses):
    idx = np.argsort(fitnesses)
    return [population[i] for i in idx[:int(ELITISM * POP_SIZE)]]

def repair_individual(x, bounds):
    x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    total = np.sum(x)
    if total > TOTAL_LIMIT:
        x = x * (TOTAL_LIMIT / total)
    return x

def run_ga(bounds, objective):
    population = init_population(bounds)
    history = []

    for gen in range(GENS):
        fitnesses = [objective(ind) for ind in population]
        best_fitness = min(fitnesses)
        history.append(best_fitness)

        new_pop = select(population, fitnesses)
        while len(new_pop) < POP_SIZE:
            parents = random.sample(new_pop, 2)
            child = crossover(parents[0], parents[1])
            if random.random() < MUT_PROB:
                child = mutate(child, bounds)
            child = repair_individual(child, bounds)
            new_pop.append(child)
        population = new_pop

    best = min(population, key=objective)
    return best, objective(best), history

# PSO Sınıf ve Fonksiyonlar
class Particle:
    def __init__(self, x, v):
        self.x = x
        self.v = v
        self.x_best = x.copy()
        self.y_best = None

def particle_swarm_optimization(f, population, seed, k_max, w=0.7, c1=1.5, c2=1.5):
    np.random.seed(seed)
    n = len(population[0].x)
    x_best = None
    y_best = float("inf")
    history = []

    for p in population:
        y = f(p.x)
        p.y_best = y
        if y < y_best:
            x_best = p.x.copy()
            y_best = y

    for k in range(k_max):
        for p in population:
            r1, r2 = np.random.rand(n), np.random.rand(n)
            p.v = w * p.v + c1 * r1 * (p.x_best - p.x) + c2 * r2 * (x_best - p.x)
            p.x += p.v
            p.x = repair_individual(p.x, bounds)
            y = f(p.x)

            if y < p.y_best:
                p.x_best = p.x.copy()
                p.y_best = y
            if y < y_best:
                x_best = p.x.copy()
                y_best = y

        history.append(y_best)

    return x_best, y_best, history

def create_population(m, lower_bound, upper_bound, seed):
    np.random.seed(seed)
    population = []
    for _ in range(m):
        x = np.random.uniform(lower_bound, upper_bound)
        x = repair_individual(x, bounds)
        v = np.random.uniform(-0.1, 0.1, len(x))  # daha küçük başlangıç hızı
        particle = Particle(x, v)
        population.append(particle)
    return population

def run_pso(bounds, objective):
    population = create_population(POP_SIZE, [b[0] for b in bounds], [b[1] for b in bounds], seed=42)
    best, best_score, history = particle_swarm_optimization(objective, population, seed=42, k_max=GENS)
    return best, best_score, history

def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")

    df["Ti"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]

    GA_real = df["alan_metrekare"].values
    GA_min, GA_max = GA_real.min(), GA_real.max()
    GA_norm = (GA_real - GA_min) / (GA_max - GA_min)

    P_real = df["Nufus"].values
    P = (P_real - P_real.min()) / (P_real.max() - P_real.min())

    Ti = df["Ti"]
    Ti = (Ti - Ti.min()) / (Ti.max() - Ti.min())

    AQI = df["Ortalama_AQI"]
    AQI = (AQI - AQI.min()) / (AQI.max() - AQI.min())

    S = W1 * AQI + W2 * Ti

    lower_bounds = np.zeros(len(GA_real))
    upper_bounds = np.minimum(GA_real / 2, PER_TOWN_LIMIT)
    global bounds
    bounds = list(zip(lower_bounds, upper_bounds))

    def objective(x):
        base = np.sum((S * P) / (GA_norm + x / GA_max + 1e-6))
        green_per_person = GA_real / (P_real + 1e-6)
        normalized_gpp = green_per_person / np.max(green_per_person)
        fairness_penalty = np.sum(x * normalized_gpp)
        return base + 0.5 * fairness_penalty

    best_ga, score_ga, history_ga = run_ga(bounds, objective)
    best_pso, score_pso, history_pso = run_pso(bounds, objective)

    df["Yeni_Yapilacak_Yesil_Alan_GA"] = best_ga
    df["Toplam_Yesil_Alan_GA"] = df["alan_metrekare"] + best_ga

    df["Yeni_Yapilacak_Yesil_Alan_PSO"] = best_pso
    df["Toplam_Yesil_Alan_PSO"] = df["alan_metrekare"] + best_pso

    print(f"\n✅ GA en iyi skor: {score_ga:.2f}")
    print(f"✅ PSO en iyi skor: {score_pso:.2f}")

    df.to_csv("optimum_yesil_alan_sonuclari.csv", index=False)
    print("\n📁 Sonuçlar 'optimum_yesil_alan_sonuclari.csv' dosyasına kaydedildi.")

    plt.figure(figsize=(10, 6))
    plt.plot(history_ga, label="GA", marker='o')
    plt.plot(history_pso, label="PSO", marker='x')
    plt.title("Yakınsama Grafiği: GA vs PSO")
    plt.xlabel("Nesil")
    plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("karsilastirma_yakinssama_grafigi.png")
    plt.show()
    print("\n📊 Karşılaştırma grafiği 'karsilastirma_yakinssama_grafigi.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()

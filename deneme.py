import pandas as pd
import numpy as np
import random

# Parametreler
W1 = 0.5
W2 = 0.5
delta = 10   # kişi başı minimum yeşil alan (şu an kullanılmıyor)
rho = 30     # kişi başı maksimum yeşil alan
TOTAL_LIMIT = 1_000_000  # toplam yapılabilecek yeşil alan sınırı

POP_SIZE = 100
GENS = 200
MUT_PROB = 0.2
ELITISM = 0.1

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
    for gen in range(GENS):
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

def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")

    df["Ti"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]
    df["Si"] = W1 * df["Ortalama_AQI"] + W2 * df["Ti"]

    GA = df["alan_metrekare"].values
    P = df["Nufus"].values
    S = df["Si"].values

    # İlçe bazında alt/üst sınırlar
    lower_bounds = np.zeros(len(GA))
    upper_bounds = np.minimum(rho * P - GA, GA / 2)
    upper_bounds = np.maximum(upper_bounds, 0)  # negatif çıkmasın
    bounds = list(zip(lower_bounds, upper_bounds))

    # Objective function + total limit penalty
    def objective(x):
        total_new_area = np.sum(x)
        penalty = 1e6 if total_new_area > TOTAL_LIMIT else 0
        return np.sum((S * P) / (GA + x + 1)) + penalty

    # GA çalıştır
    best_solution, best_score = run_ga(bounds, objective)

    # Sonuçları kaydet
    df["Yeni_Yapilacak_Yesil_Alan"] = best_solution
    df["Toplam_Yesil_Alan"] = df["alan_metrekare"] + best_solution

    print(f"\n✅ En iyi skor: {best_score:.2f}")
    print(f"📏 Toplam yeni yapılan yeşil alan: {np.sum(best_solution):,.2f} m²")
    print(df[["ILCE", "Yeni_Yapilacak_Yesil_Alan", "Toplam_Yesil_Alan"]])

    df.to_csv("optimum_yesil_alan_sonuclari.csv", index=False)
    print("\n📁 Sonuçlar 'optimum_yesil_alan_sonuclari.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()

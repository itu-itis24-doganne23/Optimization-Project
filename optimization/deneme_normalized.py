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

def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")

    # Toplu taşıma puanı
    df["Ti"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]

    # Normalizasyonlar
    GA_real = df["alan_metrekare"].values
    GA_min, GA_max = GA_real.min(), GA_real.max()
    GA_norm = (GA_real - GA_min) / (GA_max - GA_min)

    P_real = df["Nufus"].values
    P = (P_real - P_real.min()) / (P_real.max() - P_real.min())

    Ti = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]
    Ti = (Ti - Ti.min()) / (Ti.max() - Ti.min())

    AQI = df["Ortalama_AQI"]
    AQI = (AQI - AQI.min()) / (AQI.max() - AQI.min())

    S = W1 * AQI + W2 * Ti

    lower_bounds = np.zeros(len(GA_real))
    upper_bounds = np.minimum(GA_real / 2, PER_TOWN_LIMIT)
    bounds = list(zip(lower_bounds, upper_bounds))

    def objective(x):
        # Performans kısmı (normalize değerlerle)
        base = np.sum((S * P) / (GA_norm + x / GA_max + 1e-6))  # Normalize edilmiş yeşil alan + yeni alan (normalize)

        # Kişi başına düşen mevcut yeşil alan (ceza için gerçek değerlerle)
        green_per_person = GA_real / (P_real + 1e-6)  # bölme hatası engeli
        normalized_gpp = green_per_person / np.max(green_per_person)  # normalize et

        # Ceza: kişi başına zaten çok düşüyorsa ve hâlâ yeni alan veriyorsak, cezalandır
        fairness_penalty = np.sum(x * normalized_gpp)

        return base + 0 * fairness_penalty  # 0.5 ceza katsayısı, denemeye açık

    best_solution, best_score, history = run_ga(bounds, objective)

    # Gerçek metrekare cinsinden yeni yeşil alanları hesaplarken normalize edilmiş GA yerine orijinal metrekareleri kullanıyoruz
    df["Yeni_Yapilacak_Yesil_Alan"] = best_solution
    df["Toplam_Yesil_Alan"] = df["alan_metrekare"] + best_solution

    print(f"\n✅ En iyi skor: {best_score:.2f}")
    print(f"📏 Toplam yeni yapılan yeşil alan: {np.sum(best_solution):,.2f} m²")
    print(df[["ILCE", "Yeni_Yapilacak_Yesil_Alan", "Toplam_Yesil_Alan"]])

    df.to_csv("optimum_yesil_alan_sonuclari.csv", index=False)
    print("\n📁 Sonuçlar 'optimum_yesil_alan_sonuclari.csv' dosyasına kaydedildi.")

    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', color='green')
    plt.title("Genetik Algoritma Yakınsama Grafiği")
    plt.xlabel("Nesil")
    plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ga_yakinssama_grafigi.png")
    plt.show()
    print("\n📊 Yakınsama grafiği 'ga_yakinssama_grafigi.png' olarak kaydedildi.")


if __name__ == "__main__":
    main()

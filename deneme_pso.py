import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parametreler
W1 = 0.5
W2 = 0.5
TOTAL_LIMIT = 3_000_000
PER_TOWN_LIMIT = 1_000_000

NUM_PARTICLES = 100
MAX_ITER = 200
W_INERTIA = 0.7
C1 = 1.5
C2 = 1.5
SEED = 42

np.random.seed(SEED)

### Yardımcı Fonksiyon ###
def repair_vector(x, bounds):
    x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    total = np.sum(x)
    if total > TOTAL_LIMIT:
        x *= TOTAL_LIMIT / total
    return x

### Objective Function ###
def objective(x, S, P, GA):
    return np.sum((S * P) / (GA + x + 1))

### PSO Sınıfı ###
class Particle:
    def __init__(self, dim, bounds):
        self.x = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
        self.v = np.random.uniform(-1000, 1000, dim)
        self.x = repair_vector(self.x, bounds)
        self.x_best = self.x.copy()
        self.y_best = float('inf')

def run_pso(bounds, S, P, GA):
    dim = len(bounds)
    swarm = [Particle(dim, bounds) for _ in range(NUM_PARTICLES)]

    g_best_x = None
    g_best_y = float('inf')
    history = []

    for k in range(MAX_ITER):
        for p in swarm:
            y = objective(p.x, S, P, GA)
            if y < p.y_best:
                p.y_best = y
                p.x_best = p.x.copy()
            if y < g_best_y:
                g_best_y = y
                g_best_x = p.x.copy()

        for p in swarm:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = C1 * r1 * (p.x_best - p.x)
            social = C2 * r2 * (g_best_x - p.x)
            p.v = W_INERTIA * p.v + cognitive + social
            p.x += p.v
            p.x = repair_vector(p.x, bounds)

        history.append(g_best_y)

    return g_best_x, g_best_y, history

### Ana Fonksiyon ###
def main():
    df = pd.read_csv("result/birlesik_ilce_verisi.csv")
    df["Ti"] = df["Minibus_Durak_Sayisi"] + df["Taksi_Durak_Sayisi"] + 2 * df["Rayli_Istasyon_Sayisi"]
    df["Si"] = W1 * df["Ortalama_AQI"] + W2 * df["Ti"]

    GA = df["alan_metrekare"].values
    P = df["Nufus"].values
    S = df["Si"].values

    lower_bounds = np.zeros(len(GA))
    upper_bounds = np.minimum(GA / 2, PER_TOWN_LIMIT)
    bounds = list(zip(lower_bounds, upper_bounds))

    best_sol, best_score, history = run_pso(bounds, S, P, GA)

    df["PSO_Yeni_Yesil_Alan"] = best_sol
    df["PSO_Toplam_Yesil_Alan"] = df["alan_metrekare"] + best_sol

    print(f"\n✅ En iyi PSO skoru: {best_score:.2f}")
    print(f"📏 Toplam yeni yapılan yeşil alan (PSO): {np.sum(best_sol):,.2f} m²")
    print(df[["ILCE", "PSO_Yeni_Yesil_Alan", "PSO_Toplam_Yesil_Alan"]])

    df.to_csv("optimum_yesil_alan_sonuclari_pso.csv", index=False)
    print("📁 Sonuçlar 'optimum_yesil_alan_sonuclari_pso.csv' dosyasına kaydedildi.")

    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', color='blue')
    plt.title("PSO Yakınsama Grafiği")
    plt.xlabel("Iterasyon")
    plt.ylabel("Amaç Fonksiyonu (Z)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pso_yakinssama_grafigi.png")
    # plt.show()
    print("📊 Yakınsama grafiği 'pso_yakinssama_grafigi.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()
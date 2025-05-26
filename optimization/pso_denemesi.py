import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parametreler
W1 = 0.5
W2 = 0.5
TOTAL_LIMIT = 3_000_000
PER_TOWN_LIMIT = 1_000_000

POP_SIZE = 100
MAX_ITER = 200
W_INERTIA = 0.7
C1 = 1.5
C2 = 1.5

def repair_individual(x, bounds):
    x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    total = np.sum(x)
    if total > TOTAL_LIMIT:
        x = x * (TOTAL_LIMIT / total)
    return x

def run_pso(bounds, objective):
    dim = len(bounds)
    particles = np.array([repair_individual(
        np.array([np.random.uniform(lb, ub) for lb, ub in bounds]), bounds) for _ in range(POP_SIZE)])
    velocities = np.zeros_like(particles)

    personal_best = np.copy(particles)
    personal_best_scores = np.array([objective(p) for p in particles])

    global_best_idx = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_idx]
    global_best_score = personal_best_scores[global_best_idx]

    history = [global_best_score]

    for _ in range(MAX_ITER):
        for i in range(POP_SIZE):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (W_INERTIA * velocities[i] +
                             C1 * r1 * (personal_best[i] - particles[i]) +
                             C2 * r2 * (global_best - particles[i]))
            particles[i] = repair_individual(particles[i] + velocities[i], bounds)

            score = objective(particles[i])
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = score

        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        history.append(global_best_score)

    return global_best, global_best_score, history

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
    bounds = list(zip(lower_bounds, upper_bounds))

    def objective(x):
        #base = np.sum((S * P) / (GA_norm + x / GA_max + 1e-6))
        #green_per_person = GA_real / (P_real + 1e-6)
        #normalized_gpp = green_per_person / np.max(green_per_person)
        #fairness_penalty = np.sum(x * normalized_gpp)
        #return base + 0 * fairness_penalty
        x_norm = x / GA_real.max()
        return np.sum((S * P) / (GA_norm + x_norm + 1e-3))

    best_solution, best_score, history = run_pso(bounds, objective)

    df["Yeni_Yapilacak_Yesil_Alan"] = best_solution
    df["Toplam_Yesil_Alan"] = df["alan_metrekare"] + best_solution

    print(f"\n✅ En iyi skor (PSO): {best_score:.2f}")
    print(f"📏 Toplam yeni yapılan yeşil alan: {np.sum(best_solution):,.2f} m²")
    print(df[["ILCE", "Yeni_Yapilacak_Yesil_Alan", "Toplam_Yesil_Alan"]])

    df.to_csv("optimum_yesil_alan_sonuclari_pso.csv", index=False)
    print("\n📁 Sonuçlar 'optimum_yesil_alan_sonuclari_pso.csv' dosyasına kaydedildi.")

    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', color='blue')
    plt.title("Parçacık Sürü Optimizasyonu Yakınsama Grafiği")
    plt.xlabel("İterasyon")
    plt.ylabel("En İyi Amaç Fonksiyonu Değeri (Z)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pso_yakinssama_grafigi.png")
    plt.show()
    print("\n📊 Yakınsama grafiği 'pso_yakinssama_grafigi.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("../outputs/one_scenario.csv")


df["kisi_basi_mevcut"] = df["alan_metrekare"] / df["Nufus"]
df["kisi_basi_ga"] = df["Yeni_Yapilacak_Yesil_Alan_GA_W1_0.5_W2_0.5"] / df["Nufus"]
df["kisi_basi_pso"] = df["Yeni_Yapilacak_Yesil_Alan_PSO_W1_0.5_W2_0.5"] / df["Nufus"]

ilceler = df["ILCE"]
kisi_basi_mevcut = df["kisi_basi_mevcut"]
kisi_basi_ga = df["kisi_basi_ga"]
kisi_basi_pso = df["kisi_basi_pso"]

# divide
orta_index = len(df) // 2

ilceler_1 = ilceler[:orta_index]
mevcut_1 = kisi_basi_mevcut[:orta_index]
ga_1 = kisi_basi_ga[:orta_index]
pso_1 = kisi_basi_pso[:orta_index]


ilceler_2 = ilceler[orta_index:]
mevcut_2 = kisi_basi_mevcut[orta_index:]
ga_2 = kisi_basi_ga[orta_index:]
pso_2 = kisi_basi_pso[orta_index:]


def plot_grouped_bar(ilceler, mevcut, ga, pso, title,save):
    x = np.arange(len(ilceler))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, mevcut, width, label="Current per Person", color="green")
    ax.bar(x, ga, width, label="PSO per Persone", color="orange")
    ax.bar(x + width, pso, width, label="PSO per Persone", color="blue")

    ax.set_ylabel("Green Area per Person (m2)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ilceler, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.show()

# first graph
plot_grouped_bar(ilceler_1, mevcut_1, ga_1, pso_1, "Green Area per Person","first_group.png")

# second graph
plot_grouped_bar(ilceler_2, mevcut_2, ga_2, pso_2, "Green Area per Person","second_group.png")

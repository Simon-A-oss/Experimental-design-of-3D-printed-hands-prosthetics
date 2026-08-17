"""
Analyse de calibration de capteurs de masse.

Lit la feuille "raw data" du fichier SensorsCalibration.xlsx et trace, pour
chaque capteur :
  1. La difference absolue  : mesure - theorique   (en N)
  2. La difference relative : (mesure - theorique) / theorique * 100  (en %)

La valeur theorique est F = m * g. La feuille contient deja cette colonne
("theoretical value"), calculee avec la masse (colonne "Masse/capteur",
supposee en grammes) et g = 9.81 m/s^2 -- verifie sur la premiere ligne non
nulle : 0.1069 kg * 9.81 = 1.0487 N, ce qui correspond a la valeur du fichier.

Usage :
    python plot_sensor_calibration.py chemin/vers/SensorsCalibration.xlsx
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------------------------

xlsx_path = sys.argv[1] if len(sys.argv) > 1 else "SensorsCalibration.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="raw data")

mass_col = "Masse/capteur"
theo_col = "theoretical value"

# Colonnes de mesure : tout sauf la masse et la valeur theorique
sensor_cols = [c for c in df.columns if c not in (mass_col, theo_col)]

# Certaines colonnes "adjusted" contiennent des cellules vides -> texte.
# On force la conversion en nombres (les cellules non convertibles deviennent NaN).
for c in sensor_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# On ignore la ligne masse = 0 (theorique = 0 -> division par zero pour le relatif)
data = df[df[mass_col] > 0].copy()

# ---------------------------------------------------------------------------
# 2. Calcul des ecarts
# ---------------------------------------------------------------------------

abs_diff = pd.DataFrame({mass_col: data[mass_col]})
rel_diff = pd.DataFrame({mass_col: data[mass_col]})

for c in sensor_cols:
    abs_diff[c] = data[c] - data[theo_col]
    rel_diff[c] = (data[c] - data[theo_col]) / data[theo_col] * 100

# Tous les capteurs bruts (C1..C6) et "adjusted" sont maintenant traces
# ensemble sur les memes graphiques. On associe une couleur par numero de
# capteur (C1, C2, ...) et on distingue raw/adjusted par le style de trait
# et le marqueur, afin que "C1" et "C1 adjusted" soient facilement reperables
# comme la meme paire tout en restant visuellement distincts.

base_sensor_names = []
for c in sensor_cols:
    base = c.replace(" adjusted", "")
    if base not in base_sensor_names:
        base_sensor_names.append(base)

cmap = plt.get_cmap("tab10")
color_of = {name: cmap(i % 10) for i, name in enumerate(base_sensor_names)}

def style_of(col):
    base = col.replace(" adjusted", "")
    is_adjusted = "adjusted" in col
    return {
        "color": color_of[base],
        "linestyle": "--" if is_adjusted else "-",
        "marker": "s" if is_adjusted else "o",
        "label": col,
    }

# ---------------------------------------------------------------------------
# 3. Trace des graphiques
# ---------------------------------------------------------------------------

def plot_group(diff_df, cols, ylabel, title, filename):
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    for c in cols:
        plt.plot(diff_df[mass_col], diff_df[c], **style_of(c))
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.xlabel("Masse (g)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)

    # Quadrillage plus fin : grille majeure + grille mineure (via ticks
    # mineurs automatiques), pour mieux lire les valeurs intermediaires.
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# --- Groupe 1 : tous les capteurs (raw + adjusted) ---
plot_group(abs_diff, sensor_cols, "Mesure - Theorique (N)",
           "Difference absolue - tous les capteurs (raw vs adjusted)",
           "abs_diff_all.png")
plot_group(rel_diff, sensor_cols, "Difference relative (%)",
           "Difference relative - tous les capteurs (raw vs adjusted)",
           "rel_diff_all.png")

# --- Groupe 2 : sans C2 brut (C2 adjusted reste affiche) ---
cols_no_c2_raw = [c for c in sensor_cols if c != "C2"]
plot_group(abs_diff, cols_no_c2_raw, "Mesure - Theorique (N)",
           "Difference absolue - sans C2 brut",
           "abs_diff_no_c2raw.png")
plot_group(rel_diff, cols_no_c2_raw, "Difference relative (%)",
           "Difference relative - sans C2 brut",
           "rel_diff_no_c2raw.png")

print("Graphiques enregistres : abs_diff_all.png, rel_diff_all.png, "
      "abs_diff_no_c2raw.png, rel_diff_no_c2raw.png")
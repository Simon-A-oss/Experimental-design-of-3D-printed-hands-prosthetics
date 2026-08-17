"""
============================================================================================================
README

ce code sert à plot les caractéristiques des élastiques orthodontiques,
pour décider lesquels utiliser pour les prothèses.

Etant donné que la fonction des elastiques est ici de rouvrir la prothèse,
les critères qui déterminent le meilleur élastique sont les suivants :
    - la force au seuil bas d'allongement (0.59 inches) doit être suffisante pour rouvrir la prothèse 
        - mais tout exces au dessus de ce seuil est inutile, donc on cherche le plus proche au dessus du seuil
        - jusqu'à un certain niveau d'usure acceptable, ce qui décale verticalement la plage acceptable vers le haut
    - la pente de la courbe force/elongation doit etre la plus faible possible
        - car tant que la force est suffisante pour rouvrir la prothèse c'est bon
        - tout exces de force rends l'utilisation de la prothèse moins agréable
============================================================================================================
"""


import matplotlib.pyplot as plt
import numpy as np

# lien commande
# https://www.fruugo.be/dental-orthodontic-rubber-bands-ortho-elastics-latex-braces-non-toxic-tool/p-331741057-731542678?language=en

# functions
def oz_to_newton(oz):
    return oz * 0.2780139

def newton_to_oz(newton):
    return newton / 0.2780139

def inches_to_cm(inches):
    return inches * 2.54

def cm_to_inches(cm):
    return cm / 2.54

# data
quarter_offset = cm_to_inches(0.589)
fiveOver16_offset = cm_to_inches(0.838)
threeOver8_offset = cm_to_inches(1.088)
quarter_35_y = np.array([0, 3.5/2, 3.5, 3.5*1.5])
quarter_35_x = np.array([0.25, 0.5, 0.75, 1.0]) + quarter_offset - 0.25 

quarter_50_y = np.array([0, 5/2, 5, 5*1.5])
quarter_50_x = np.array([0.25, 0.5, 0.75, 1.0])+ quarter_offset - 0.25

quarter_65_y = np.array([0, 6.5/2, 6.5, 6.5*1.5])
quarter_65_x = np.array([0.25, 0.5, 0.75, 1.0])+ quarter_offset - 0.25

five_over16_35_y = np.array([0, 3.5/2, 3.5, 3.5*1.5])
five_over16_35_x = np.array([5/16, 10/16, 15/16, 20/16]) + fiveOver16_offset - 5/16

five_over16_50_y = np.array([0, 5/2, 5, 5*1.5])
five_over16_50_x = np.array([5/16, 10/16, 15/16, 20/16]) + fiveOver16_offset - 5/16

five_over16_65_y = np.array([0, 6.5/2, 6.5, 6.5*1.5])
five_over16_65_x = np.array([5/16, 10/16, 15/16, 20/16]) + fiveOver16_offset - 5/16

three_over8_35_y = np.array([0, 3.5/2, 3.5, 3.5*1.5])
three_over8_35_x = np.array([3/8, 6/8, 9/8, 12/8]) + threeOver8_offset - 3/8

three_over8_50_y = np.array([0, 5/2, 5, 5*1.5])
three_over8_50_x = np.array([3/8, 6/8, 9/8, 12/8]) + threeOver8_offset - 3/8

three_over8_65_y = np.array([0, 6.5/2, 6.5, 6.5*1.5])
three_over8_65_x = np.array([3/8, 6/8, 9/8, 12/8]) + threeOver8_offset - 3/8

three_over8_45_y = np.array([0, 4.5/2, 4.5, 4.5*1.5])
three_over8_45_x = np.array([3/8, 6/8, 9/8, 12/8]) + threeOver8_offset - 3/8


# plots

# plot 1 : force vs elongation
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(inches_to_cm(quarter_35_x), oz_to_newton(quarter_35_y), label='1/4-3.5', marker='o')
ax.plot(inches_to_cm(quarter_50_x), oz_to_newton(quarter_50_y), label='1/4-5', marker='o')
ax.plot(inches_to_cm(quarter_65_x), oz_to_newton(quarter_65_y), label='1/4-6.5', marker='o')
ax.plot(inches_to_cm(five_over16_35_x), oz_to_newton(five_over16_35_y), label='5/16-3.5', marker='o')
ax.plot(inches_to_cm(five_over16_50_x), oz_to_newton(five_over16_50_y), label='5/16-5', marker='o')
ax.plot(inches_to_cm(five_over16_65_x), oz_to_newton(five_over16_65_y), label='5/16-6.5', marker='o')
ax.plot(inches_to_cm(three_over8_35_x), oz_to_newton(three_over8_35_y), label='3/8-3.5', marker='o')
ax.plot(inches_to_cm(three_over8_50_x), oz_to_newton(three_over8_50_y), label='3/8-5', marker='o')
ax.plot(inches_to_cm(three_over8_65_x), oz_to_newton(three_over8_65_y), label='3/8-6.5', marker='o')
ax.plot(inches_to_cm(three_over8_45_x), oz_to_newton(three_over8_45_y), label='3/8-4.5', marker='o')
ax.axvline(x=inches_to_cm(1), color='red', linestyle='--', label='current max elongation')
ax.axvline(x=inches_to_cm(0.59), color='red', linestyle='--', label='current min elongation')
ax.set_title('Force vs. Elongation for Different Elastic Bands')
ax.set_xlabel('Length (cm)')
ax.set_ylabel('Force (N)')
ax.legend()
ax.grid()
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)  # Adjust y-axis limit to better fit the data
secxaxis = ax.secondary_xaxis('top', functions=(cm_to_inches, inches_to_cm))
secxaxis.set_xlabel('Length (inches)')
secyaxis = ax.secondary_yaxis('right', functions=(newton_to_oz, oz_to_newton))
secyaxis.set_ylabel('Force (oz)')

# plot 2 : force vs elongation ajusted to 3.5 oz at 0.59 inches
displacements = [0, 0.213, 0.33, -0.425, -0.16, -0.016, -0.855, -0.617, -0.535, -0.36]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(inches_to_cm(quarter_35_x)+cm_to_inches(displacements[0]), oz_to_newton(quarter_35_y), label='1/4-3.5', marker='o')
ax.plot(inches_to_cm(quarter_50_x+cm_to_inches(displacements[1])), oz_to_newton(quarter_50_y), label='1/4-5', marker='o')
ax.plot(inches_to_cm(quarter_65_x+cm_to_inches(displacements[2])), oz_to_newton(quarter_65_y), label='1/4-6.5', marker='o')
ax.plot(inches_to_cm(five_over16_35_x+cm_to_inches(displacements[3])), oz_to_newton(five_over16_35_y), label='5/16-3.5', marker='o')
ax.plot(inches_to_cm(five_over16_50_x+cm_to_inches(displacements[4])), oz_to_newton(five_over16_50_y), label='5/16-5', marker='o')
ax.plot(inches_to_cm(five_over16_65_x+cm_to_inches(displacements[5])), oz_to_newton(five_over16_65_y), label='5/16-6.5', marker='o')
ax.plot(inches_to_cm(three_over8_35_x+cm_to_inches(displacements[6])), oz_to_newton(three_over8_35_y), label='3/8-3.5', marker='o')
ax.plot(inches_to_cm(three_over8_45_x+cm_to_inches(displacements[7])), oz_to_newton(three_over8_45_y), label='3/8-4.5', marker='o')
ax.plot(inches_to_cm(three_over8_50_x+cm_to_inches(displacements[8])), oz_to_newton(three_over8_50_y), label='3/8-5', marker='o')
ax.plot(inches_to_cm(three_over8_65_x+cm_to_inches(displacements[9])), oz_to_newton(three_over8_65_y), label='3/8-6.5', marker='o')
ax.axvline(x=1.3, color='red', linestyle='--', label='current min elongation')
ax.axvline(x=inches_to_cm(1), color='red', linestyle='--', label='current max elongation')

ax.set_title('Force vs. Elongation adjusted to 3.5 oz at 0.59 inches')
ax.set_xlabel('Length (cm)')
ax.set_ylabel('Force (N)')
ax.legend()
ax.grid()
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)  # Adjust y-axis limit to better fit the data
secxaxis = ax.secondary_xaxis('top', functions=(cm_to_inches, inches_to_cm))
secxaxis.set_xlabel('Length (inches)')
secyaxis = ax.secondary_yaxis('right', functions=(newton_to_oz, oz_to_newton))
secyaxis.set_ylabel('Force (oz)')

#plot 3: bar chart of displacement required to reach 3.5 oz at 0.59 inches
labels = ['1/4-3.5', '1/4-5', '1/4-6.5', '5/16-3.5', '5/16-5', '5/16-6.5', '3/8-3.5', '3/8-4.5', '3/8-5', '3/8-6.5']
adjustments = [-displacements[i] for i in range(len(displacements))]  # invert the sign to show the required displacement
plt.figure(figsize=(10, 6))
plt.bar(labels, adjustments, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8', 'C9'])
plt.axhline(0.3, color='red', linestyle='--', label='max possible adjustment')
plt.title('adjustment Required to Reach 3.5 oz at 0.59 inches')
plt.xlabel('Elastic Band')
plt.ylabel('Displacement (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
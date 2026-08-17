"""
============================================================================================================
README
Ce code sert à tout et nimporte quoi, je l'utilise pour tester des fonctions,
vérifier comment fonctionnent des structures, ...

============================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import random
from os import listdir
from os.path import isfile, join



# test to see the linearity of the time taken to process frames
x = np.array([0, 120, 300, 800, 1650])
y1 = np.array([0, 0.9937356000300497, 2.089298899983987, 4.952952999970876, 11.041988900047727])*6
y2 = np.array([0, 0.869533299934119, 2.2457110000541434, 5.761476200073957, 10.538234099978581])*6
y3 = np.array([0, 0.6482028000755236, 1.5090839999029413, 3.3481761999428272, 7.7235726999351755])*6
y4 = np.array([0, 0.5457869000001665, 1.2062468000003719, 3.1570518000007723, 6.386874000001626])*6
y5 = np.array([0, 0.551607100002002, 1.249007200007327, 3.1493370000098366, 6.543105499993544])*6
y6 = np.array([0, 0.7647050000086892, 1.8691092999943066, 4.8162121000059415, 9.714221200003522])


plt.plot(x, y1, marker='o', label='v1 - 1 finger per run')
plt.plot(x, y2, marker='o', label='v2 - 1 finger per run') 
plt.plot(x, y3, marker='o', label='y2.2 - 1 finger per run')
plt.plot(x, y4, marker='o', label='V2.3 - 1 finger per run')
plt.plot(x, y5, marker='o', label='V3 - 1 finger per run')
plt.plot(x, y6, marker='o', label='V3 - 6 fingers at once')
plt.title('Time taken to process frames vs number of frames')   
plt.xlabel('Number of frames')
plt.ylabel('Time taken (seconds)')
plt.xticks(x)  # set x-ticks to be the same as the x values
plt.grid()
plt.legend()
plt.show() #
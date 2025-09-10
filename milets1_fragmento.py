# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 19:43:41 2025

@author: Carolanp.array([r])

"""

"""
CLASE 21/8 OTRAS COSITAS, DFT
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

N=8
n=np.arange(N)
xx = 3 * np.sin(2 * np.pi* n/ 2) + 4 
yy = xx[-1::-1][:4]  

print("xx:", xx)
print("yy:", yy)
"""

import numpy as np
import matplotlib.pyplot as plt

N = 8
X = np.zeros(N, dtype=np.complex128)
n = np.arange(N)
x = 3 * np.sin(n * np.pi / 2) + 4

# Calcular la DFT manualmente
for k in range(N):
    for n in range(N):
        X[k] += x[n] * np.exp(-1j * k * 2 * np.pi * n / N)

print("Coeficientes DFT:")
print(X)

# Graficar el espectro
plt.figure(1)
markerline, stemlines, baseline = plt.stem(np.arange(N), np.abs(X))  # abs para graficar el módulo
plt.setp(markerline, color='red')     # puntitos
plt.setp(stemlines, color='red')      # palitos
plt.setp(baseline, color='blue')  # base
plt.title("DFT")
plt.xlabel("Índice k")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()
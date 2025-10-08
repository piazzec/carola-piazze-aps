# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 22:32:28 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 1000      # frecuencia de muestreo
N = 1000       # muestras totales
f = 50         # frecuencia de la senoidal
a0 = 2         # amplitud
ruido_amp = 0.5  # amplitud del ruido
r = 10         # cantidad de columnas (para la "matriz")

# Vector de tiempo
t = np.arange(N) / fs

# Señal seno con ruido
senal = a0 * np.sin(2 * np.pi * f * t) + ruido_amp * np.random.randn(N)

# Reorganizar en matriz (r columnas)
M = senal.reshape(-1, r)

# Graficar la señal (ya en el tiempo, no la FFT)
plt.figure(figsize=(10,4))
plt.plot(t, senal, label="Seno con ruido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal senoidal con ruido (dominio del tiempo)")
plt.legend()
plt.grid()
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 22:28:22 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 1000        # frecuencia de muestreo (Hz)
N = 1000         # cantidad de muestras
f = 5           # frecuencia de la senoidal (Hz)
a0 = 2           # amplitud
ruido_amp = 0.5  # amplitud del ruido

# Vector de tiempo
t = np.arange(N) / fs

# Señal: seno + ruido
senal = a0 * np.sin(2 * np.pi * f * t) + ruido_amp * np.random.randn(N)

# Graficar
plt.figure(figsize=(10,4))
plt.plot(t, senal, label="Seno con ruido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal en el dominio del tiempo")
plt.legend()
plt.grid()
plt.show()

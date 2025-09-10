# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:17:46 2025

@author: Carola
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

M = 8
x = np.zeros(M)
x[3:5] = 1
y=np.zeros(M)

y[3]=1

M=8
x[0:3]=1 #esto hace que de cero a tres valga 1
Rxx = sig.correlate(x,y)
Rxx1=sig.convolve(x,y)

plt.figure(figsize=(12,5))
plt.stem(x)   # sin el argumento extra
plt.show()


plt.figure(figsize=(12,5))
plt.plot(Rxx, 'x:')
plt.show()
plt.plot

# Calcular la correlación cruzada (usando correlate de scipy.signal)
rxy = sig.correlate(x, y, mode='full')  # Correlación cruzada
convxy = sig.convolve(x, y, mode='full')  # Convolución

# Graficar
plt.figure(1)
plt.plot(x, 'x:', label='x')  # Señal x
plt.plot(y, 'x:', label='y')  # Señal y
# plt.plot(rxy, 'o-', label='rxy')  # Correlación cruzada (comentada por ahora)
plt.plot(convxy, '*:', label='convxy')  # Convolución
plt.legend()  # Corrección: legend() en lugar de tegend()
plt.show()


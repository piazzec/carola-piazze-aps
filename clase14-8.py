# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:19:28 2025

@author: Carola
"""

"""
clase 14/8/2025
creo que estamos tratando de encontrar una cantidad de muestras que puedo tomar en un período ??
para F0=1 me quedan N muestras
para F0=2 me quedan N/2 muestras
entonces F0= Fs/2
a donde vemos que se pierde la nocion de una senoidal?
el módulo de la transformada de fourier de una senoidal son dos deltas
"""
import numpy as np
import matplotlib.pyplot as plt

def funcion_seno(amplitud, dc, frec, fase, cantmuestras, fs):
    ts = 1/fs  # periodo de muestreo
    # tiempo de muestreo según cantidad de muestras
    tt = np.arange(0, cantmuestras*ts, ts)
    # convertir fase a radianes si viene en grados
    fase_rad = np.deg2rad(fase)
    xx = amplitud * np.sin(2 * np.pi * frec * tt + fase_rad) + dc
    return tt, xx

# ejemplo de uso
# %matplotlib qt PARA QUE HAGA EL GRÁFICO AFUERA EN OTRA PESTAÑA
tt, xx = funcion_seno(3, 2, 1000, 0, 1000, 1000)

plt.figure()
plt.title("Onda sinusoidal")
plt.plot(tt, xx)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#con esto se puede hacer el tc0, nyquist, que le pasa a la señal cuando cambio nro de muestras, frecuencia, etc. 

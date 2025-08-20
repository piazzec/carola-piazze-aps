# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 21:16:46 2025

@author: Carola
"""

"""arrancamos con el tc1 POTENCIA, AUTOCORRELACION, ORTOGONALIDAD (1)
(1)TENGO QUE GRAFICAR UN COSENO Y UN ENO Y CAMBIAR LAS FRECUENCIAS Y ESO PARA VER SI SON ORTOGONALES
(2)la autocorrelación? supuestamente la explicó el otro tipo. si le pongo las dos secuencias a python lo hace solo

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
tt, xx = funcion_seno(3, 2, 999, 0, 1000, 1000)
suma=1/1000 *np.sum(xx**2) #esto es la suma que me tenia que dar amplitud al cuadrado/2 y me dio bien!!
energia=1/1000*np.var(xx)

plt.figure()
plt.title("Onda sinusoidal")
plt.plot(tt, xx)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#np.sum(x**2) me hace la suma de todo al cuadrado

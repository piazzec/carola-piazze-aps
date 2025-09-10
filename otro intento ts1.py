# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 19:49:11 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt

def funcion_seno(amplitud, dc, frec, fase, nmuestras, fs):
   
    tt = (np.arange(nmuestras) / fs).reshape(-1, 1)      
    xx = (dc + amplitud * np.sin(2 * np.pi * frec * tt + fase)).reshape(-1, 1)  
    return tt, xx
t1,x1=funcion_seno(1,0,2000,np.pi/4,1000,1000)
t2,x2=funcion_seno(1,0,1000,np.pi/4,1000,1000)
x_aux=x1*x2

plt.subplot(2,2,1)
plt.plot(t1, x1, label='Señal senoidal de 2kHz')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

# Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.

plt.subplot(2,2,2)
plt.plot(t2, x2, label='Señal modulada con otra señal de la mitad de la f')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(t2, x_aux, label='recorto al 75% de la señal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

plt.legend()
#plt.layout()
plt.show()
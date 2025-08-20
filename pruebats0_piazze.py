# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:52:40 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt

def funcion_seno(amplitud=1, dc=0.5, frec=0, fase=np.pi/4, nmuestras=1000, fs=1000):
    """Con esta función genero una señal senoidal muestreada según los parámetros correspondientes:
    Amplitud máxima (amplitud), valor medio (dc), frecuencia en Hz (frec), fase en radianes, cantidad de muestras (nmuestras), y frecuencia de muestreo en Hz (fs).
    Esta función devuelve tt
    ts = 1/fs  # periodo de muestreo
    # tiempo de muestreo según cantidad de muestras tt: vector de tiempos [s] y xx: señal muestreada [V].
    """
    tt = (np.arange(nmuestras) / fs).reshape(-1, 1)      # Vector columna Nx1
    xx = (dc + amplitud * np.sin(2 * np.pi * frec * tt + fase)).reshape(-1, 1)  # Señal Nx1
    return tt, xx

# PARTE UNO: ejemplo simple de uso
tt, xx = funcion_seno(frec=5, nmuestras=1000, fs=1000)
#la grafico primero (sola)
plt.figure(figsize=(10,4))
plt.plot(tt, xx, color="black")
plt.title("Senoidal de prueba (5 Hz)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()

# PARTE DOS: grafico de varias frecuencias (en un solo coso con un subplot)
frecuencias = [500, 999, 1001, 2001]  # Hz
plt.figure(figsize=(12, 8))  # Tamaño de la figura

for i, freq in enumerate(frecuencias):
    tt, xx = funcion_seno(frec=freq, nmuestras=4000, fs=50000)
    #le aumenté el numero de muestras a una cantidad un poco exhuberante pero me estaba quedando muy picuda, así queda bien suave
    plt.subplot(2, 2, i+1)  # subplot 2 filas, 2 columnas
    plt.plot(tt, xx, color="magenta")
    plt.xlim(0, 0.005)
    plt.title(f"Senoidal {freq} Hz")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.grid(True)
    

plt.tight_layout()
plt.show() 

# PARTE TRES: otra señal (elegí hacer una cuadrada porque la he usado en prácticas de laboratorio)

def funcion_cuadrada(amplitud=1, dc=0, frec=1, nmuestras=1000, fs=1000):
    tt = (np.arange(nmuestras) / fs).reshape(-1, 1)
    # sign(sin(...)) devuelve +1 cuando el seno >=0 y -1 cuando es <0 
    # eso del sign me lo tiró chat porque yo no ubicaba esa función en python
    xx = (dc + amplitud * np.sign(np.sin(2 * np.pi * frec * tt))).reshape(-1, 1)
    return tt, xx

# Ejemplo señal cuadrada de 50 Hz
tt, xx = funcion_cuadrada(amplitud=1, dc=0, frec=50, nmuestras=2000, fs=50000)

plt.figure(figsize=(10,4))
plt.plot(tt, xx, color="green") 
plt.title("Señal cuadrada (creada con sign)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()
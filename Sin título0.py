# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:27:51 2025

@author: Carola
"""

''' tengo que hallar y[n] con cada una de mis señales x del otro tp, luego transformar para despejar h de la convolución (y=h*x)
una vez que encuentro h[n] puedo calcular la salida y[n] para cualquier otra entrada
como me dice que son causales, yn depende solo de xk, y la salida es igual a la entrada
Sistema LTI (Lineal e Invariante en el tiempo): definido por su respuesta al impulso 
h[n]. La salida para cualquier entrada causal viene por convolución:
'''
'''
Un sistema LTI (Lineal e Invariante en el Tiempo) puede modelarse mediante ecuaciones en diferencias con coeficientes constantes. La ecuación dada:
representa un sistema recursivo (IIR) donde la salida actual depende de entradas actuales, pasadas y salidas pasadas.

La respuesta al impulso 
h[n] de un sistema LTI es la salida cuando la entrada es un impulso unitario 
δ[n]. Para sistemas IIR, la respuesta al impulso tiene duración infinita, pero en la práctica se aproxima con una longitud finita.

La salida de un sistema LTI puede calcularse mediante la convolución entre la entrada y la respuesta al impulso:
Para señales discretas, se implementa con np.convolve.
    '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import math


fs = 60000      
Ts = 1 / fs        

#copio señales ts1
t_seno = np.linspace(0, 0.002, int(fs*0.002), endpoint=False)   # 2 ms
t_cuadrada = np.linspace(0, 0.005, int(fs*0.005), endpoint=False)  # 5 ms
t_pulso = np.linspace(0, 0.02, int(fs*0.02), endpoint=False)     # 20 ms

frec = 2000.0  # Hz
x1 = np.sin(2*np.pi*frec*t_seno)                      # seno 2 kHz
x2 = 2*np.sin(2*np.pi*frec*t_seno + np.pi/2)          # A=2, desfase pi/2
frec_mod = 1000.0
m = 0.8
x3 = (1 + m*np.sin(2*np.pi*frec_mod*t_seno)) * np.sin(2*np.pi*frec*t_seno)  # AM
A = np.max(np.abs(x1))
x1_clipped = np.clip(x1, -0.75*A, 0.75*A)
frec_cuadrada = 4000.0
x_cuadrada = np.sign(np.sin(2*np.pi*frec_cuadrada*t_cuadrada))

#defino las ecuaciones de energía y potencia, después las voy a calcular y printear para cada función según corresponda
def calc_potencia(x): #(para periódicas)
    return np.mean(x**2)

def calc_energia(x, Ts): #(para finitas)
    return np.sum(x**2) * Ts
# guardo las señales en un diccionario para iterar
señales = {
    'x1_seno': (x1, t_seno),
    'x2_seno_A2_desfase': (x2, t_seno),
    'x3_AM': (x3, t_seno),
    'x1_clipped': (x1_clipped, t_seno),
    'x_cuadrada': (x_cuadrada, t_cuadrada),

}

#tener en cuenta que como es causal todo lo que pase antes (ejemplo y[n-1]) cuando n es 0 debe ser cero
def LTI_1(x):
    N = len(x)          # cantidad de muestras
    y = np.zeros(N)     # vector salida

    # caso base n=0
    y[0] = 0.03 * x[0]

    # caso base n=1
    if N > 1:  # para no romper si la señal es muy corta
        y[1] = 0.03*x[1] + 0.05*x[0] + 1.5*y[0]

    # para n >= 2
    for n in range(2, N):
        y[n] = (0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2]
                 + 1.5*y[n-1] - 0.5*y[n-2])
    return y


# =============================
# PROBAR EL SISTEMA CON UNA SEÑAL
# =============================

# recorro las señales definidas
for nombre, (x, t) in señales.items():
    y = LTI_1(x)  # aplico la ecuación en diferencias

    # gráfico comparando entrada y salida
    plt.figure(figsize=(8,4))
    plt.plot(t*1000, x, label="Entrada " + nombre)
    plt.plot(t*1000, y, label="Salida y[n]")
    plt.xlabel("Tiempo [ms]")
    plt.title("Sistema LTI aplicado a " + nombre)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcular energía o potencia
    if nombre == 'x_cuadrada' or nombre.startswith('x1_seno'):
        pot = calc_potencia(y)
        print(f"Potencia de salida para {nombre}: {pot:.4f}")
    else:
        ener = calc_energia(y, Ts)
        print(f"Energía de salida para {nombre}: {ener:.4f}")

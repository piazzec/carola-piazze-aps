#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:38:59 2025

@author: victoria24
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 


#Plantilla de diseño 
#vamos a hacer un pasabajo

wp = 1 #frecuencia de corte en rad/s (es lo mismo que de paso)
ws = 5 #frecuencia de stop (rad/s)
 
alpha_p = 1 #atenuacion maxima a la wp, alpha max o pérdidas en banda de paso (dB)
alpha_s = 40 #atenuacion minima a la ws, alpha min o minima atenuacion requerida en banda de paso (dB)

#Aprox modulo 
f_aprox = 'butter'
# f_aprox = 'cheby1'
# f_aprox = 'cheby2'
# f_aprox = 'cauer'

b, a = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=True, ftype=f_aprox, output='ba')
#devuelve coeficientes del polinomio 

# %%
 
# w, h = signal.freqs(b, a, worN=np.logspace(1,6,1000)) 
#espacio logaritmicamente espaciado. entre 10¹, 10⁶ y va a tomar 1000 puntos entre ambos

w, h = signal.freqs(b, a) # calcula la respuesta en frecuencia del filtro

phase = np.unwrap(np.angle(h)) #unwrap es para que las discontinuidades evitables sean evitadas

#Retardo de grupo = -delta phi / delta w
gd = -np.diff(phase) / np.diff(w)

#Polos y ceros
z, p, k = signal.tf2zpk(b, a)   

#graficos

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(phase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%

# T1(s)

# Coeficientes
b, a = sig.zpk2tf(z_a, p_a, k_a)

# Frecuencia logarítmica
w = np.logspace(-1, 2, 1000)

# Respuesta en frecuencia
w, h = sig.freqs(b, a, w)

# Fase sin saltos
phase = np.unwrap(np.angle(h))

# Retardo de grupo
gd = -np.diff(phase) / np.diff(w)

# --- Gráficas ---
plt.figure(figsize=(12,5))

# Magnitud
plt.subplot(1,2,1)
plt.semilogx(w, 20*np.log10(np.abs(h)))
plt.title('T1(s) - Respuesta en Magnitud')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(1,2,2)
plt.semilogx(w, np.degrees(phase))
plt.title('T1(s) - Respuesta en Fase')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')


#%% Respuesta en frecuencia estilo Bode para T2(s)

b, a = sig.zpk2tf(z_b, p_b, k_b)
w, h = sig.freqs(b, a, w)
phase = np.unwrap(np.angle(h))
gd = -np.diff(phase) / np.diff(w)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.semilogx(w, 20*np.log10(np.abs(h)))
plt.title('T2(s) - Respuesta en Magnitud')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

plt.subplot(1,2,2)
plt.semilogx(w, np.degrees(phase))
plt.title('T2(s) - Respuesta en Fase')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')


#%% Respuesta en frecuencia estilo Bode para T3(s)

b, a = sig.zpk2tf(z_c, p_c, k_c)
w, h = sig.freqs(b, a, w)
phase = np.unwrap(np.angle(h))
gd = -np.diff(phase) / np.diff(w)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.semilogx(w, 20*np.log10(np.abs(h)))*


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:07:19 2025

@author: Carola
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio



fs=2
fir_win_pm =[1,1]
fir_win_pm2=[1,0,1]
fir_win_pm3=[1,0,0,0,0,1]


# --- Cálculo de fase y retardo de grupo ---

w1, h1 = sig.freqz(b=fir_win_pm, fs=fs)
w2, h2 = sig.freqz(b=fir_win_pm2, fs=fs)
w3, h3 = sig.freqz(b=fir_win_pm3, fs=fs)


fase1 = np.unwrap(np.angle(h1))
fase2 = np.unwrap(np.angle(h2))
fase3 = np.unwrap(np.angle(h3))

w_rad1 = w1 / (fs / 2) * np.pi
w_rad2 = w2 / (fs / 2) * np.pi
w_rad3 = w3 / (fs / 2) * np.pi

gd1 = -np.diff(fase1) / np.diff(w_rad1)
gd2 = -np.diff(fase2) / np.diff(w_rad2)
gd3 = -np.diff(fase3) / np.diff(w_rad3)

# --- Gráficos ---
plt.figure(figsize=(10,8))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w1, 20*np.log10(abs(h1)), label='[1,1]')
plt.plot(w2, 20*np.log10(abs(h2)), label='[1,0,1]')
plt.plot(w3, 20*np.log10(abs(h3)), label='[1,0,0,0,0,1]')
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(3,1,2)
plt.plot(w1, fase1, label='[1,1]')
plt.plot(w2, fase2, label='[1,0,1]')
plt.plot(w3, fase3, label='[1,0,0,0,0,1]')
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w1[1:], gd1, label='[1,1]')
plt.plot(w2[1:], gd2, label='[1,0,1]')
plt.plot(w3[1:], gd3, label='[1,0,0,0,0,1]')
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

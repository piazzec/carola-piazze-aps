# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:37:36 2025

@author: Carola
"""

# -*- coding: utf-8 -*-
"""
Diseño comparativo de filtros FIR: firwin2, firls y remez (Parks–McClellan)
@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000  # Frecuencia de muestreo [Hz]

# --- Especificaciones del filtro ---
wp = [0.8, 35]   # Frecuencias de paso [Hz]
ws = [0.1, 35.7] # Frecuencias de stop [Hz]

cant_coef = 2001
retardo = (cant_coef - 1) // 2

# ============================================================
# --- Definición de bandas y respuestas ---
# ============================================================

# FIRWIN2 y FIRLS aceptan 6 puntos (inicio y fin de cada zona)
bands_firwin = np.array([0, ws[0], wp[0], wp[1], ws[1], fs/2])
gain_firwin  = [0, 0, 1, 1, 0, 0]

# REMEZ (o FIRPM) requiere 3 bandas → 6 puntos + 3 amplitudes
bands_remez  = [0, ws[0], wp[0], wp[1], ws[1], fs/2]
desired_remez = [0, 1, 0]  # una ganancia por banda

# ============================================================
# --- FIR con firwin2 ---
# ============================================================
fir_win = signal.firwin2(numtaps=cant_coef, freq=bands_firwin, gain=gain_firwin, window='boxcar', fs=fs)

# ============================================================
# --- FIR con firls ---
# ============================================================
fir_ls = signal.firls(numtaps=cant_coef, bands=bands_firwin, desired=gain_firwin, fs=fs)

# ============================================================
# --- FIR con remez (equivalente a firpm) ---
# ============================================================
fir_pm = signal.remez(numtaps=cant_coef-1, bands=bands_remez, desired=desired_remez, fs=fs)

# ============================================================
# --- Función para analizar filtros ---
# ============================================================
def analizar_filtro(coefs, titulo):
    w, h = signal.freqz(coefs, worN=2048, fs=fs)
    fase = np.unwrap(np.angle(h))
    w_rad = w / (fs/2) * np.pi
    gd = -np.diff(fase) / np.diff(w_rad)

    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(w, 20*np.log10(np.abs(h)))
    plt.title(f'Respuesta en Magnitud — {titulo}')
    plt.ylabel('|H(f)| [dB]')
    plt.grid(True, which='both', ls=':')

    plt.subplot(3, 1, 2)
    plt.plot(w, fase)
    plt.title('Fase')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')

    plt.subplot(3, 1, 3)
    plt.plot(w[:-1], gd)
    plt.title('Retardo de grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [muestras]')
    plt.grid(True, which='both', ls=':')
    
    plt.tight_layout()
    plt.show()

# ============================================================
# --- Visualización comparativa ---
# ============================================================
analizar_filtro(fir_win, "FIR — firwin2")
analizar_filtro(fir_ls,  "FIR — firls (Least Squares)")
analizar_filtro(fir_pm,  "FIR — remez (Parks–McClellan)")

# ============================================================
# --- Polos y ceros ---
# ============================================================
z, p, k = signal.tf2zpk(fir_ls, 1)
plt.figure()
plt.scatter(np.real(z), np.imag(z), c='r', label='Ceros')
plt.scatter(np.real(p), np.imag(p), c='b', label='Polos')
circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.title('Polos y Ceros — firls')
plt.xlabel('Re{z}')
plt.ylabel('Im{z}')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

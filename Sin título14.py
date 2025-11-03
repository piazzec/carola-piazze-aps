# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:50:50 2025

@author: Carola
"""

##algunas celdas si funcionan

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio

# %% =========================================
# PARÁMETROS GENERALES
# =============================================
fs = 1000  # Hz, frecuencia de muestreo

# IIR: bandas de paso y stop
wp_iir = [1, 35]      # Hz, banda de paso
ws_iir = [0.01, 40]   # Hz, banda de stop

# Atenuaciones
alpha_p = 1   # dB, máximo en banda de paso
alpha_s = 40  # dB, mínimo en banda de stop

# FIR: bandas de paso y stop
wp_fir = [0.8, 35]
ws_fir = [0.1, 40]
numtaps_fir = 100  # cantidad de coeficientes FIR

# %% =========================================
# DISEÑO DE FILTROS IIR
# =============================================
iir_filters = {}
for ftype in ['cauer', 'butter', 'cheby1', 'cheby2']:
    sos = signal.iirdesign(wp=wp_iir, ws=ws_iir,gpass=alpha_p, gstop=alpha_s, analog=False, ftype=ftype,
        output='sos', fs=fs
    )
    iir_filters[ftype] = sos

# %% =========================================
# DISEÑO DE FILTRO FIR
# =============================================
frecuencias = np.sort(np.concatenate(([0, fs/2], wp_fir, ws_fir, [fs/2])))
deseado = [0, 0, 1, 1, 0, 0]  # magnitud deseada en cada frecuencia
fir_coef = signal.firwin2(numtaps=numtaps_fir, freq=frecuencias, gain=deseado, fs=fs)
f_aprox_fir = 'FIR Hamming'

# Retardo aproximado del FIR
demora = (len(fir_coef) - 1) // 2

# Respuesta en frecuencia FIR
w_fir, h_fir = signal.freqz(fir_coef, worN=np.logspace(-2, 2, 1000), fs=fs)
phase_fir = np.unwrap(np.angle(h_fir))
w_rad_fir = w_fir / (fs/2) * np.pi
gd_fir = -np.diff(phase_fir) / np.diff(w_rad_fir)

# Polos y ceros FIR (sólo ceros, a=1)
z_fir, p_fir, k_fir = signal.sos2zpk(signal.tf2sos(b=fir_coef, a=1))

# %% =========================================
# LECTURA DE ECG
# =============================================
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()  # asegurar vector 1D
N = len(ecg_one_lead)

# %% =========================================
# FILTRADO ECG CON IIR
# =============================================
ecg_filt_iir = {}
for key, sos in iir_filters.items():
    ecg_filt_iir[key] = signal.sosfilt(sos, ecg_one_lead)

# FILTRADO ECG CON FIR
ecg_filt_fir = signal.lfilter(fir_coef, 1, ecg_one_lead)

# %% =========================================
# PLOT ECG FILTRADO
# =============================================
plt.figure(figsize=(12, 4))
plt.plot(ecg_filt_iir['butter'][:5000], label='Butterworth')
plt.plot(ecg_filt_iir['cauer'][:5000], label='Cauer')
plt.plot(ecg_filt_iir['cheby1'][:5000], label='Cheby1')
plt.plot(ecg_filt_iir['cheby2'][:5000], label='Cheby2')
plt.plot(ecg_filt_fir[:5000], label='FIR Hamming', linestyle='--')
plt.title('ECG Filtrado (Primeras 5000 muestras)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()

# %% =========================================
# REGIONES DE INTERÉS CON Y SIN RUIDO
# =============================================
# Con ruido
regs_interes_ruido = [
    [4000, 5500],
    [10000, 11000]
]

for ii in regs_interes_ruido:
    zoom_region = np.arange(max(0, ii[0]), min(N, ii[1]), dtype=int)
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=2)
    plt.plot(zoom_region, ecg_filt_iir['butter'][zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_fir[zoom_region + demora], label='FIR Hamming')
    plt.title(f'Región con ruido: {ii[0]}-{ii[1]} muestras')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

# Sin ruido (en minutos convertidos a muestras)
regs_interes_sin_ruido = [
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs
]

for ii in regs_interes_sin_ruido:
    zoom_region = np.arange(max(0, int(ii[0])), min(N, int(ii[1])), dtype=int)
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=2)
    plt.plot(zoom_region, ecg_filt_iir['butter'][zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_fir[zoom_region + demora], label='FIR Hamming')
    plt.title(f'Región sin ruido: {int(ii[0])}-{int(ii[1])} muestras')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

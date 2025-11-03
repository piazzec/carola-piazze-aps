# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:23:47 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio

#diseño y aplicación de filtros digitales pasabanda a una señal de ecg
#diseñamos varios filtros distintos y los aplicamos  ala misma señal ruidosa para comparar cual limpia mejor


#diseño de pasabanda digital, parametrosd e la plantilla
fs = 1000 #[Hz]
wp = [1, 35] #comienzo y fin banda de paso PASABANDA: La región que quiero mantener sin afectar
ws = [0.01, 40]  #banda de stop [Hz] STOPBAND: reguin que quiero atenuar el ruido

#cuanto más cercanos esten wp y ws más alto será el orden del filtro (más coeficientes)
# wp = [100, 135] #comienzo y fin banda de paso
# ws = [99, 140]  #banda de stop [Hz]

#transicion en pocos hz implica orden mas elevado del polinomio 
 
alpha_p = 1 #atenuacion maxima a la wp, alpha max o pérdidas en banda de paso (dB)
#cuánto se permite que se deforme la amplitud de la señal útil (1 dB = muy poco).
alpha_s = 40 #atenuacion minima a la ws, alpha min o minima atenuacion requerida en banda de paso (dB)
#cuánto se atenúan las señales fuera de la banda (mínimo 40 dB = fuerte rechazo).


#Aproximaciones modulo 
f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
#devuelve coeficientes del polinomio 
#signal iir design diseña un filtro iir digital a partir de los requisitos de la plantilla
#sos Second Order Sections
f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby1'
mi_sos_cheby1 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby2'
mi_sos_cheby2 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

# %%

# #w, h = signal.freqz_sos(mi_sos, worN=np.logspace(-2, 1.9, 1000), fs=fs)
# #espacio logaritmicamente espaciado. entre 10⁻2, 10⁶ y va a tomar 1000 puntos entre ambos

# w, h = signal.freqz_sos(mi_sos, fs=fs) # calcula la respuesta en frecuencia del filtro
# #w frecuencias donde evalua y h respuesta

# phase = np.unwrap(np.angle(h)) #unwrap es para que las discontinuidades evitables sean evitadas

# #Retardo de grupo = -delta phi / delta w
# w_rad = w / (fs/2) * np.pi 
# gd = -np.diff(phase) / np.diff(w_rad)

# #Polos y ceros
# z, p, k = signal.sos2zpk(mi_sos)   

# #graficos

# # --- Gráficas ---
# plt.figure()

# # Magnitud
# plt.subplot(2,2,1)
# plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
# plt.title('Respuesta en Magnitud')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|H(jω)| [dB]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Fase
# plt.subplot(2,2,2)
# plt.plot(w, phase, label = f_aprox)
# plt.title('Fase')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Retardo de grupo
# plt.subplot(2,2,3)
# plt.plot(w[:-1], gd, label = f_aprox)
# plt.title('Retardo de Grupo')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('τg [(#muestras)]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Diagrama de polos y ceros
# plt.subplot(2,2,4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Diagrama de Polos y Ceros (plano s)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

#%% 23-10-25
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

ecg_filt_butter = signal.sosfilt(mi_sos_butter, ecg_one_lead)
ecg_filt_cauer = signal.sosfilt(mi_sos_cauer, ecg_one_lead)
ecg_filt_cheby1 = signal.sosfilt(mi_sos_cheby1, ecg_one_lead)
ecg_filt_cheby2 = signal.sosfilt(mi_sos_cheby2, ecg_one_lead)

plt.figure()

plt.plot(ecg_filt_butter[:50000], label = 'butter')
plt.plot(ecg_filt_cauer[:50000], label = 'cauer')
plt.plot(ecg_filt_cheby1[:50000], label = 'cheby1')
plt.plot(ecg_filt_cheby2[:50000], label = 'cheby2')

plt.legend()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:57:31 2025

@author: Carola
"""

#filtro digital
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio #esta es la que lee archivos


##PLANTILLA

# pasabanda digital para ECG

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


#FILTROS IIR CON DISTINTAS APROXIMACIONES

# Aprox módulo

filtros = {
    'Butterworth': signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                    analog=False, ftype='butter', output='sos', fs=fs),
    #respuesta suave, no tiene ripple, buena fase relativa
    'Cauer (Elliptic)': signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                         analog=False, ftype='ellip', output='sos', fs=fs),
    #ripple en ambas bandas, transición más empinada
    'Chebyshev I': signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                    analog=False, ftype='cheby1', output='sos', fs=fs),
    #ripple en la banda de paso a cambio de una transición más pronunciada                                
    'Chebyshev II': signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                     analog=False, ftype='cheby2', output='sos', fs=fs)
    #ripple en la banda de stop
          }
                                    
#iirdesign diseñacon las especificaciones de la plantilla que armé (wp, ws, alphas)
#sos= Second Order Sections, en vez de darme un polinomio único con coeficientes a yb me devuelve una matriz de filas, cada fila con los coeficientes de un biquad
#(b0 b1 b2 a0 a1 a2). EL filtro es la cascada (multiplicación) de todos esos biquads


#RESPUESTAS EN FRECUENCIA
#es cómo el filtro modifica amplitud y fase de cada frecuencia
#sosfreqz evalúa la resp en un filtro sos, worN me genera puntos de frecuencia en escala logarítmica

plt.figure(figsize=(10, 6))
w = np.linspace(0, fs/2, 1000)

for nombre, sos in filtros.items():
    w_f, h = signal.sosfreqz(sos, worN=w, fs=fs)
    plt.plot(w_f, 20 * np.log10(np.maximum(abs(h), 1e-8)), label=nombre)

plt.ylim(-100, 5)
plt.title('Respuesta en frecuencia de los filtros IIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

#POLOS, FASE Y RETARDO DE GRUPO con cauer

mi_sos = filtros['Cauer (Elliptic)']
w, h = signal.freqz_sos(mi_sos, worN=w, fs=fs)
fase = np.unwrap(np.angle(h))  # unwrap evita discontinuidades en la fase

# Retardo de grupo
w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad)

# Polos y ceros
z, p, k = signal.sos2zpk(mi_sos)

plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Respuesta en Magnitud (Cauer)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

plt.subplot(2, 2, 2)
plt.plot(w, fase)
plt.title('Fase (Cauer)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')

plt.subplot(2, 2, 3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo (Cauer)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

plt.subplot(2, 2, 4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (Cauer)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#LECTURA Y FILTRADO DE SEÑAL ECG
#para listar las variables que hay en el archivo
# sio.whosmat('ECG_TP4.mat')


mat_struct = sio.loadmat('./ECG_TP4.mat') ## ojo con esto que no lo lee!! revisar
ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

# FILTRADO DEL ECG (filtfilt mantiene la alineación de los picos R)
ecg_filtrados = {}
for nombre, sos in filtros.items():
    ecg_filtrados[nombre] = signal.sosfiltfilt(sos, ecg_one_lead)
#sosfilt filtra adelante , introduce retardo de fase
#sosfiltfilt filtra hacia adelante y hacia atrás, elimina toda la distorsión de fase, duplica el orden del filtro atenúa más la banda de stop)
#grafico comparativ (con zoom)
plt.figure(figsize=(12, 6))
plt.plot(ecg_one_lead[:50000], label='ECG original', color='k', alpha=0.5, linewidth=1)
for nombre, sig in ecg_filtrados.items():
    plt.plot(sig[:50000], label=nombre)
plt.title('ECG filtrado con diferentes aproximaciones IIR')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [a.u.]')
plt.legend()
plt.tight_layout()
plt.show()

#Diagrama de polos y ceros comparativo

plt.figure(figsize=(10, 10))
theta = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

for nombre, sos in filtros.items():
    z, p, k = signal.sos2zpk(sos)
    plt.plot(np.real(p), np.imag(p), 'x', label=f'Polos {nombre}')
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', fillstyle='none', label=f'Ceros {nombre}')

plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Diagrama de Polos y Ceros de los Filtros IIR')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

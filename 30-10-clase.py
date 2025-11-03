# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:14:44 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio

# Plantilla de diseño 
# pasabanda digital 

fs = 1000 #[Hz]
wp = [1, 35] #comienzo y fin banda de paso
ws = [0.01, 40]  #banda de stop [Hz]

# wp = [100, 135] #comienzo y fin banda de paso
# ws = [99, 140]  #banda de stop [Hz]

#transicion en pocos hz implica orden mas elevado del polinomio 
 
alpha_p = 1 #atenuacion maxima a la wp, alpha max o pérdidas en banda de paso (dB)
alpha_s = 40 #atenuacion minima a la ws, alpha min o minima atenuacion requerida en banda de paso (dB)

#Aprox modulo 

f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
#devuelve coeficientes del polinomio 

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

#%% diseño de fiir (reciclo varias cosas de lo anterior)
wp = [0.8, 35] #comienzo y fin banda de paso
ws = [0.1, 40]  #banda de stop [Hz]

frecuencias=np.sort(np.concatenate(((0,fs/2),wp,ws, fs/2) ))
deseado= [0,0,1,1,0,0] #respuesta deseada del filtro en esa frecuencia, quiero que valga 1, que deje pasar en ese sector
cant_coef=100 #numtaps es coeficientes, no confundir con orden
fir_win_hamming=signal.firwin2(numtaps=cant_coef, freq=frecuencias, gain=deseado, fs=fs)
#hamming es la predeterminada
mi_sos=mi_sos_cauer
w, h=signal.freqz(mi_sos, worN=np.logspace(-2,2, 1000), fs=fs)
 # calcula la respuesta en frecuencia del filtro
# #w frecuencias donde evalua y h respuesta

phase = np.unwrap(np.angle(h)) #unwrap es para que las discontinuidades evitables sean evitadas

#Retardo de grupo = -delta phi / delta w
w_rad = w / (fs/2) * np.pi 
gd = -np.diff(phase) / np.diff(w_rad)

#Polos y ceros
z, p, k = signal.sos2zpk(signal.tf2sos(b=fir_win_hamming, a=1))   

#graficos

# --- Gráficas ---
plt.figure()

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.plot(w, phase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [(#muestras)]')
plt.grid(True, which='both', ls=':')
plt.legend()

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
#iirdesign me devuelve los coeficientes a y b

#en fiir no necesito usar sos porque no es recursivo
#en el fiir los coeficientes b son la respuesta al impulso, puramente
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


###################################
#%% Regiones de interés con ruido #
###################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure(1)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
#%% Regiones de interés sin ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure(2)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:36:42 2025

@author: Carola
"""

#a ver si por dios funciona (algunas cosas sí funcionan!)


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio

# Plantilla de diseño 
# pasabanda digital  

fs = 1000 #[Hz]
wp = [1, 35] #comienzo y fin banda de paso
ws = [0.01, 40]  #banda de stop [Hz]

#transicion en pocos hz implica orden mas elevado del polinomio 
 
alpha_p = 1  # pérdidas en banda de paso (dB)
alpha_s = 40 # mínima atenuación en banda de stop (dB)

# Aprox módulo IIR
f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                 analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby1'
mi_sos_cheby1 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                 analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby2'
mi_sos_cheby2 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                                 analog=False, ftype=f_aprox, output='sos', fs=fs)

#%% diseño de FIR (reciclo varias cosas de lo anterior)
wp = [0.8, 35] #comienzo y fin banda de paso
ws = [0.1, 40]  #banda de stop [Hz]

# vector de frecuencias ordenado y sin duplicados
frecuencias = [0, ws[0], wp[0], wp[1], ws[1], fs/2]
deseado     = [0,     0,    1,    1,    0,   0]

cant_coef = 100 #numtaps es coeficientes, no confundir con orden
b_fir = signal.firwin2(numtaps=cant_coef, freq=frecuencias, gain=deseado, fs=fs)

# Respuesta en frecuencia FIR
w, h = signal.freqz(b_fir, worN=np.logspace(-2,2,1000), fs=fs)
phase = np.unwrap(np.angle(h)) #unwrap evita saltos de fase
w_rad = w / (fs/2) * np.pi 
gd = -np.diff(phase) / np.diff(w_rad)

# Polos y ceros FIR (plano-z)
z, p, k = signal.tf2zpk(b_fir, 1)   

# Gráficos FIR
plt.figure()

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label='FIR')
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.plot(w, phase, label='FIR')
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label='FIR')
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Diagrama de ceros
plt.subplot(2,2,4)
plt.plot(np.real(z), np.imag(z), 'o', markersize=8, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Ceros (plano z)')
plt.xlabel('Re{z}')
plt.ylabel('Im{z}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% Lectura de ECG
fs_ecg = 1000 # Hz

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

# Filtrado con IIR
ecg_filt_butter = signal.sosfilt(mi_sos_butter, ecg_one_lead)
ecg_filt_cauer  = signal.sosfilt(mi_sos_cauer,  ecg_one_lead)
ecg_filt_cheby1 = signal.sosfilt(mi_sos_cheby1, ecg_one_lead)
ecg_filt_cheby2 = signal.sosfilt(mi_sos_cheby2, ecg_one_lead)

# Filtrado con FIR
ecg_fir = signal.lfilter(b_fir, [1], ecg_one_lead)
demora = (cant_coef-1)//2

plt.figure()
plt.plot(ecg_filt_butter[:50000], label = 'butter')
plt.plot(ecg_filt_cauer[:50000],  label = 'cauer')
plt.plot(ecg_filt_cheby1[:50000], label = 'cheby1')
plt.plot(ecg_filt_cheby2[:50000], label = 'cheby2')
plt.plot(ecg_fir[:50000], label = 'FIR')
plt.legend()
plt.show()

###################################
#%% Regiones de interés con ruido #
###################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10000, 11000], # muestras
        )
 
for ii in regs_interes:
   
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype=int)
   
    plt.figure(1)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_fir[zoom_region+demora], label='FIR Window')
   
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
        (np.array([5, 5.2]) *60*fs).astype(int), 
        (np.array([12, 12.4]) *60*fs).astype(int), 
        (np.array([15, 15.2]) *60*fs).astype(int), 
        )
 
for ii in regs_interes:
   
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype=int)
   
    plt.figure(2)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_fir[zoom_region+demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()